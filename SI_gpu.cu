/*
  Sampling and importance module implementation on GPU
 */
//#define _GNU_SOURCE /* for sched_getcpu() */
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sched.h> /* for sched_getcpu() */
// POSIX shared memory --- Do not forget -lrt
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
// System V semaphore
#include <sys/ipc.h> /* ftok() */
#include <sys/types.h>
#include <sys/sem.h>
extern "C" {
#include "semun.h" /* defines union semun */
/*
  declaration of currTime()
  Implementation found in TLPI P194
*/
#include "curr_time.h"
#include "tlpi_hdr.h"
#include "util.h"
}
/*
  cuRAND Device API headers
  Device functions are called in the kernel to generate random numbers
  which can be used directly without involving global memory
*/
#include <cuda.h>
#include <curand_kernel.h>

/* model specific headers */
#ifdef triv
#include "model_triv.h"
#endif
#ifdef VAR
#include "model_VAR.h"
#endif
#ifdef EKD
#include "model_EKD.h"
#endif


#define CUDA_CALL(x) do { if((x) != cudaSuccess) {              \
            printf("Error at %s:%d\n",__FILE__,__LINE__);       \
            return EXIT_FAILURE;}} while(0)

/* cuRAND state setup kernel */
__global__ void setup_kernel(curandState *state)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    /* Each thread gets same seed, a different sequence
       number, no offset */
    curand_init(1234, id, 0, &state[id]);
}
/*
  sampling and importance kernel
  TODO:
  - The layout of particles in the global memory does not lead to coalesced access.
    We need transpose the memory either on the main process or the gpu process
 */
#ifdef triv
__global__ void SI_triv(curandState *state,
                        state_t *particles,
                        double *weights,
                        ob_t *ob){
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    curandState localState = state[pid];
    /* propagation */
    state_t p_new[STATE_DIM];
    p_new[0] = particles[pid*STATE_DIM  ] + 1.0 + curand_uniform(&localState)-.5;
    p_new[1] = particles[pid*STATE_DIM+1] + 1.0 + curand_uniform(&localState)-.5;
    /* importance */
    double dis = sqrt((p_new[0]-ob[0])*(p_new[0]-ob[0]) +
                      (p_new[1]-ob[1])*(p_new[1]-ob[1]));
    /* particle and weight update */
    weights[pid] = 1.0 / (dis+1.0);
    particles[pid*STATE_DIM  ] = p_new[0];
    particles[pid*STATE_DIM+1] = p_new[1];
    state[pid] = localState;
}
#elif defined(VAR)
__global__ void SI_VAR(curandState *state,
                       state_t *particles,
                       double *weights,
                       ob_t *ob, input_t *input){
    int pid = threadIdx.x + blockDim.x * blockIdx.x;
    curandState localState = state[pid];

    float2 rnn;
    state_t *p_old = &particles[pid*STATE_DIM];
    state_t p_new[STATE_DIM];
    /* Reconstruct inputs */
    double  Vram, P;
    Vram = input[in_Vramc];
    P = (Vc + Ri * p_old[8]) * p_old[8];
    /* Sample process uncertainties */
    rnn = curand_normal2(&localState);
    double dI = dt * sigmaI * rnn.x;
    double dVram = dt * sigmaVram * rnn.y;
    rnn = curand_normal2(&localState);
    double dmu = sqrt(dt) * sigmamur * rnn.x;
    double dhe = sqrt(dt) * sigmahe * rnn.y;
    /* Differential equations */
    double deltadot, Gdot, Medot;
    double Scldot, Smrdot;
    deltadot = alphar*Cdd/p_old[0] - 4.0*Cdp*p_old[4]*P/pi/De/De/hm;
    Gdot = a0 * (-alphar*Csd/p_old[0] + 4.0*Csp*p_old[4]*P/hm/pi/De/De)-Vram;
    Medot = pi*rhor*De*De*alphar*Csd/4.0/p_old[0] - rhor*Csp*p_old[4]*P/hm;
     if (Medot >= epsilon)
     	Medot = epsilon;
    Scldot = -Acl*(p_old[6]-Scl0)+Bdeltacl*(p_old[0]-delta0)+Bicl*(p_old[8]-I0)+
        Bmucl*(p_old[4]-mu0)+Bhecl*(p_old[5]-phe0);
    Smrdot = -Amr*(p_old[7]-Smr0)+Bdeltamr*(p_old[0]-delta0)+Bimr*(p_old[8]-I0)+
        Bmumr*(p_old[4]-mu0)+Bhemr*(p_old[5]-phe0);
    /* propagation */
    // Delta: electrode thermal boundary layer
    p_new[0] = p_old[0] + deltadot*dt + G11*dI;
    if (p_new[0] < epsilon)
     	p_new[0] = epsilon;
    // G: electrode gap
    p_new[1] = p_old[1] + Gdot*dt+ G21*dI - dVram;
    if (p_new[1] < epsilon)
     	p_new[1] = epsilon;
    // Xram: ram position
    p_new[2] = p_old[2] + dt*Vram + dVram;
    // Me: electrode mass
    p_new[3] = p_old[3] + Medot*dt + G41*dI;
    if (p_new[3] < epsilon)
     	p_new[3] = epsilon;
    // mu: melting efficiency
    p_new[4] = p_old[4] + dmu;
    if (p_new[4] < epsilon)
     	p_new[4] = epsilon;
    // phe: helium pressure
    p_new[5] = p_old[5] + dhe;
    if (p_new[5] < epsilon)
     	p_new[5] = epsilon;
    // Scl: centerline pool depth and Smr: mid-radius pool depth
    p_new[6] = p_old[6] + Scldot*dt + Bicl*dI;
    if (p_new[6] < epsilon)
     	p_new[6] = epsilon;
    p_new[7] = p_old[7] + Smrdot*dt + Bimr*dI;
    if (p_new[7] < epsilon)
     	p_new[7] = epsilon;
    double mI = input[in_Ic] + (input[in_Ic] - p_old[8])*exp(-dt/1.0);
    p_new[8] = mI + dI;

    /* Importance */
    double w = 0.0;
    double y_G    = p_new[1];
    double y_Xram = p_new[2];
    double y_I    = p_new[8];
    double y_Me   = p_new[3];
    double y_Scl  = p_new[6];
    double y_Smr  = p_new[7];
    double y_phe  = p_new[5];
    double expo =
        (ob[ob_G]    - y_G)    * (ob[ob_G]    - y_G)    / sigmaG/sigmaG +
        (ob[ob_Xram] - y_Xram) * (ob[ob_Xram] - y_Xram) / sigmaPos/sigmaPos +
        (ob[ob_I]    - y_I)    * (ob[ob_I]    - y_I)    / sigmaImeas/sigmaImeas +
        (ob[ob_Me]   - y_Me)   * (ob[ob_Me]   - y_Me)   / sigmaLC/sigmaLC +
        (ob[ob_Scl]  - y_Scl)  * (ob[ob_Scl]  - y_Scl)  / sigmaCL/sigmaCL +
        (ob[ob_Smr]  - y_Smr)  * (ob[ob_Smr]  - y_Smr)  / sigmaMR/sigmaMR +
        (ob[ob_phe]  - y_phe)  * (ob[ob_phe]  - y_phe)  / sigmahemeas/sigmahemeas;
    w = exp(-0.5 * expo);

    /* particle and weight update */
    weights[pid] = w + 1e-99;
    for (int st = 0; st < STATE_DIM; st ++)
        particles[pid*STATE_DIM+st] = p_new[st];
    state[pid] = localState;
}
#else
#endif
/*
  particle sub-population is saved in shared memory
  The number of particles is passed from my parent with a signal,
  which also notifies the readiness of the sub-population in the shared memory.
 */
int main(int argc, char ** argv)
{
    LOG1("GPU process starting on CPU %d", sched_getcpu());
    if (argc < 2) errExit("N not specified");
    int N = getInt(argv[1], 0, "total number of particles");

    /*
      Open the semaphore set created by the main process
      Note that we do not need to deal with the race condition issue because
      the main process created and initialized the semaphore before calling fork.
     */
    key_t key = ftok("./main", 'x');
    if (-1 == key) errExit("ftok");
    int semid = semget(key, 1, S_IRUSR | S_IWUSR);
    if (-1 == semid) errExit("semget");
    LOG2("Opened the semaphore set %d (value = %d)", semid, GETSEM);
    struct sembuf sops;

    /*
      We are getting the sub-particle population and the number of particles
      from the shared memory prepared by the main process.
     */
    int fd_pp, fd_gpu_n, fd_w, fd_ob, fd_in;
    size_t p_sz = STATE_DIM * sizeof(state_t);
    size_t pp_sz = N * p_sz;
    size_t gpu_n_sz = sizeof (int);
    size_t w_sz  = sizeof(double) * N;
    size_t ob_sz = sizeof(ob_t) * OB_DIM;
    size_t in_sz = sizeof(input_t) * INPUT_DIM;
    void *addr_pp, *addr_n, *addr_w, *addr_ob, *addr_in;
    open_shm(fd_gpu_n, "/shm_gpu_n", gpu_n_sz, addr_n);
    open_shm(fd_pp,    "/shm_pp",    pp_sz,    addr_pp);
    open_shm(fd_w,     "/shm_w",     w_sz,     addr_w);
    open_shm(fd_ob,    "/shm_ob",    ob_sz,    addr_ob);
    open_shm(fd_in,    "/shm_in",    in_sz,    addr_in);

    /* Check GPU status */
    int device;
    struct cudaDeviceProp properties;
    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaGetDeviceProperties(&properties,device));
    LOG1("Hello from %s", properties.name);

#ifdef triv
    const unsigned int threads = 64;
#else // VAR and EKD
    const unsigned int threads = 128;
#endif
    unsigned int blocks  = (N+threads-1) / threads;

    /* Initialize curand */
    curandState *devStates;
    CUDA_CALL(cudaMalloc((void **)&devStates, N*sizeof(curandState)));
    setup_kernel<<<blocks, threads>>>(devStates);
    CUDA_CALL(cudaDeviceSynchronize());
    LOG("Finished setting up curand states");

    state_t *particles = (state_t*) addr_pp;
    state_t *devParticles;
    double *weights = (double*)addr_w;
    double *devWeights;
    ob_t *obs = (ob_t*)addr_ob;
    input_t *inputs = (input_t*)addr_in;
    CUDA_CALL(cudaMalloc((void**)&devParticles, pp_sz));
    CUDA_CALL(cudaMalloc((void**)&devWeights, w_sz));
    ob_t *devObs;
    CUDA_CALL(cudaMalloc((void**)&devObs, ob_sz));
    input_t *devInputs;
    CUDA_CALL(cudaMalloc((void**)&devInputs, in_sz));


    ////////////////////////////////////////////////
    // start of current iteration
    ////////////////////////////////////////////////
    /*
      Before starting the iteration, decrement the semaphore.
      This should set the semaphore to 0 and the main process should
      know that we are ready.
     */
    semop0(-1);
    int iter = 1;
    do{
        LOG1(">>>>>>>>>> Iteration %d starts: <<<<<<<<<<", iter);
        /*
          Sync between the main process by waiting for the semaphore to be 2
          and decrement the semaphore by 1

          And this marks the beginning of the current iteration on GPU

          Note that, if N_gpu == 0, the semaphore value will never be increased.
          Therefore, we are going to be blocked here until being killed by the
          main process.
        */
        semop0(-1);
        LOG1("Particles are ready (sema value = %d)", GETSEM);
        /* First get N_gpu for gpu */
        int N_gpu = *((int*) addr_n);
        if (0 == N_gpu){
            LOG("No particles assigned");
            goto finish_update;
        }
        /* Then copy N_gpu particles to device */
        CUDA_CALL(cudaMemcpy(devParticles, particles, p_sz*N_gpu, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(devObs, obs, ob_sz, cudaMemcpyHostToDevice));
        CUDA_CALL(cudaMemcpy(devInputs, inputs, in_sz, cudaMemcpyHostToDevice));
        /*
          Update the particles from the shared memory
        */
        LOG1("Updating particles 0 ~ %d on gpu: ", (N_gpu-1));
        printOb(obs);
        /*
          Pay attention that we are working on shared memory pointers here.
          Make sure we do not overwrite the part of memory that does not belong
          to us. This is done by making sure only N_gpu particles are processed
          and only the first N_gpu weights are updated.
         */
        blocks = (N_gpu+threads-1) / threads;
#ifdef triv
        SI_triv<<<blocks,threads>>>(devStates, devParticles, devWeights, devObs);
#elif defined(VAR)
        SI_VAR<<<blocks, threads>>>(devStates, devParticles, devWeights, devObs, devInputs);
#elif defined(EKD)
        LOG("EKD model not implemented yet!");
#else
        LOG("Unknown model!");
        exit(0);
#endif
        /* At last, copy result back to host */
        CUDA_CALL(cudaMemcpy(particles, devParticles, p_sz*N_gpu, cudaMemcpyDeviceToHost));
        CUDA_CALL(cudaMemcpy(weights, devWeights, sizeof(double)*N_gpu, cudaMemcpyDeviceToHost));
        cudaDeviceSynchronize();
        LOG("Finished updating particles");
finish_update:
        /*
          Now it is the time to decrement the semaphore so that the main process can
          see value 0 in the semaphore
        */
        semop0(-1);
        LOG2("Decrement semaphore %d again (value = %d)", semid, GETSEM);
        LOG1(">>>>>>>>>> Iteration %d ends: <<<<<<<<<<", iter);
        iter ++;
    }while(1);
    /////////////////////////////////////////////////
    // end of current iteration
    /////////////////////////////////////////////////
cleanup:
    LOG("I am all done. Bye yo");

    CUDA_CALL(cudaFree(devParticles));
    CUDA_CALL(cudaFree(devWeights));
    CUDA_CALL(cudaFree(devObs));
    CUDA_CALL(cudaFree(devInputs));
    exit(0);
}
