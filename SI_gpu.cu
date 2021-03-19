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
__global__ void SI(curandState *state,
                   state_t *particles,
                   double *weights,
                   obs_t *ob){
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
    int fd_pp, fd_gpu_n, fd_w, fd_ob;
    size_t p_sz = STATE_DIM * sizeof(state_t);
    size_t pp_sz = N * p_sz;
    size_t gpu_n_sz = sizeof (int);
    size_t w_sz  = sizeof(double) * N;
    size_t ob_sz = sizeof(obs_t) * OBS_DIM;
    void *addr_pp, *addr_n, *addr_w, *addr_ob;
    open_shm(fd_gpu_n, "/shm_gpu_n", gpu_n_sz, addr_n);
    open_shm(fd_pp,    "/shm_pp",    pp_sz,    addr_pp);
    open_shm(fd_w,     "/shm_w",     w_sz,     addr_w);
    open_shm(fd_ob,    "/shm_ob",    ob_sz,    addr_ob);

    /* Check GPU status */
    int device;
    struct cudaDeviceProp properties;
    CUDA_CALL(cudaGetDevice(&device));
    CUDA_CALL(cudaGetDeviceProperties(&properties,device));
    LOG1("Hello from %s", properties.name);

    const unsigned int threads = 64;
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
    obs_t *obs = (obs_t*)addr_ob;
    obs_t *devObs;
    CUDA_CALL(cudaMalloc((void**)&devParticles, pp_sz));
    CUDA_CALL(cudaMalloc((void**)&devWeights, w_sz));
    CUDA_CALL(cudaMalloc((void**)&devObs, ob_sz));


    ////////////////////////////////////////////////
    // start of current iteration
    ////////////////////////////////////////////////
    int iter = 1;
    do{
        LOG1(">>>>>>>>>> Iteration %d starts: <<<<<<<<<<", iter);
        /*
          Sync between the main process by waiting for the semaphore to be 2
          and decrement the semaphore by 1

          And this marks the beginning of the current iteration on GPU
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
        //printp(0, 15);
        /*
          Update the particles from the shared memory
        */
        LOG1("Updating particles 0 ~ %d on gpu: ", (N_gpu-1));
        LOG2("Current observations: %lf, %lf", obs[0], obs[1]);
        /*
          Pay attention that we are working on shared memory pointers here.
          Make sure we do not overwrite the part of memory that does not belong
          to us. This is done by making sure only N_gpu particles are processed
          and only the first N_gpu weights are updated.
         */
        blocks = (N_gpu+threads-1) / threads;
        SI<<<blocks,threads>>>(devStates, devParticles, devWeights, devObs);
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
    LOG("I am all done. Bye yo");

    CUDA_CALL(cudaFree(devParticles));
    CUDA_CALL(cudaFree(devWeights));
    CUDA_CALL(cudaFree(devObs));
    exit(0);
}
