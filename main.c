//#define _POSIX_C_SOURCE 199309
#define _GNU_SOURCE /* for sched_getcpu() */
#include <sys/wait.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <time.h>
#include <math.h> /* for sqrt() */
#include <sched.h> /* for sched_getcpu() */
#include <signal.h> /* for kill() */
// POSIX shared memory --- Don't forget -lrt
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
// System V semaphore
#include <sys/ipc.h> /* ftok() */
#include <sys/types.h>
#include <sys/sem.h>
#include "semun.h" /* defines union semun */
/*
  declaration of currTime()
  Implementation found in TLPI P194
 */
#include "curr_time.h"
#include "tlpi_hdr.h"
#include "util.h"
// Model specific headers
#ifdef triv
#include "model_triv.h"
#endif
#ifdef VAR
#include "model_VAR.h"
#endif
#ifdef EKD
#include "model_EKD.h"
#endif


/*
  Let's try to use semaphore to sync between the main and gpuProcess.
  - main creates a semaphore with initial value of 0.
  - main starts to copy particle population to the shared memory.
  - When done, add 2 to the semaphore.
  - On the gpuProcess, decrement the semaphore by 1. (If the shared memory
  is not ready, the semaphore will have value 0. And the gpuProcess will
  block until the shared memory is ready.)
  - When shared memory is ready. gpuProcess update the particles.
  - After it is finished, decrement the semaphore again.
  - At the same time, after adding 2 to the semaphore, the main process
  waits for the semaphore to be zero, which is an indication that the
  gpuProcess has finished updating the particles.
  - When the semaphore becomes zero again, the main process can proceed.

  Since we need to add 2 to the semaphore and wait for the semaphore to be
  zero, System V semaphore is the only option. Though the interface of the
  POSIX semaphore is much simpler, we have to go through the complexity of
  System V semaphore for the above functionality.
*/

/*
  Compute the optimal number of particles for gpu process
 */
int particle_partition(int n, int argc, char ** argv);
/*
  get observations for the current time step
 */
void getObs(obs_t *ob, int t);

/* Print the particle with the max weight */
void printMax(state_t *particles, double * weight, int start, int end);

/* prescan the weight array */
double prescan(double *weights, int N);

int main(int argc, char ** argv)
{
    /*
      get N, N_gpu, and t_max from the command line
      default values:
      N = 5000
      N_gpu = 1024
      t_max = 101
     */
    int N = 5000, N_gpu, t_max = 101;
    if (argc > 1)
        N = getInt(argv[1], 0, "total number of particles");
    if (argc > 3)
        t_max = getInt(argv[3], 0, "number of iterations");
    int width = (int)log10(N) + 1;

    size_t p_sz  = STATE_DIM * sizeof(state_t);
    size_t pp_sz = p_sz * N;
    size_t w_sz  = sizeof(double) * N;
    size_t ob_sz = sizeof(obs_t) * OBS_DIM;

    struct timespec tp_start, tp_end;
    LOG2("starting with %d particles on CPU %d", N, sched_getcpu());
    LOG2("STATE_DIM = %d  state size: %ld bytes", STATE_DIM, p_sz);

    /*
      Three shared memory objects are needed for each CU:
      1. particle populations
      2. number of particles
      3. particle weights

      And we create the shared memory objects before creating the child processes
      so that when the child processes starts, the shared memory should already be
      opened.
    */
    int fd_pp, fd_w, fd_gpu_n, fd_ob;
    void *addr_pp, *addr_w, *addr_n, *addr_ob;
    create_shm(fd_pp,    "/shm_pp",   pp_sz,        addr_pp);
    create_shm(fd_w,     "/shm_w",    w_sz,         addr_w);
    create_shm(fd_gpu_n, "/shm_gpu_n", sizeof(int), addr_n);
    create_shm(fd_ob,    "/shm_ob",   ob_sz,        addr_ob);
    LOG("Opened and mmapped shared memory for particles, weights, gpu_n, and obs");

    state_t* particles = (state_t*) addr_pp;
    double* weights = (double*)addr_w;
    obs_t* ob = (obs_t*)addr_ob;

    /*
      System V semaphore

      1. Get a key (consider it as a filename for the semaphore set)
      2. Create a semaphore set (consider semid as a file descriptor)
      with 1 semaphore
      3. Initialize the semaphore's value to 0

      Note that IPC_EXCL is not specified here for the same reason as O_EXCL is
      not specified in shm_open()
    */
    key_t key = ftok("./main", 'x');
    if (-1 == key) errExit("ftok");
    int semid = semget(key, 1, IPC_CREAT | S_IRUSR | S_IWUSR);
    if (-1 == semid) errExit("semget");
    union semun arg;
    arg.val = 0;
    if (-1 == semctl(semid, 0, SETVAL, arg)) errExit("semctl");
    LOG2("Opened and initialized the semaphore set %d (value = %d)", semid, GETSEM);
    struct sembuf sops;

    /*
      Initialize particle population
      This is a model-dependent operation, hence the prefix mdl_
      There is an agreement between the model developer the framework that the
      layout of the population is as follows:

      Each row of the "matrix" represents one particle

      This is because it is easier to partition the particle population
      horizontally.
     */
    TIME_START;
    mdl_init_particles(particles, N);
    TIME_END;
    LOG1("Initializing particles done in %lf seconds", ELAPSEDTIME);

    //LOG("Particles:");
    //printp(0, 10);
    //printp((N-10), N);

    /*
      Creating a child process to run particle filter on the GPU

      fork() is said to be not as fast as vfork() or clone().
      However, given that the creation of new processes for each CU is a
      one-time operation, the overhead of fork() should be negligible.
     */
    pid_t gpuPid;
    char N_arg[100];
    sprintf(N_arg, "%d", N);
    switch(gpuPid = fork()){
    case -1:
        errExit("fork");
    case 0:
        execl("./SI_gpu", "SI_gpu", N_arg, (char*)NULL);
        errExit("execl");
        break;
    default:
        break;
    }
    LOG1("main process running on CPU %d", sched_getcpu());

    //////////////////////////////////////////////////////////
    // Iteration t starts
    /////////////////////////////////////////////////////////
    double w, ess;
    int t = 1, N_cpu;
    state_t p_new[STATE_DIM];
    double time_inc = 0.0;
    do {
        LOG1(">>>>>>>>>> Iteration %d starts: <<<<<<<<<<", t);
        /*
          Assign a number of particles to the gpu process
          This means the first N_gpu rows of the particle population will be processed
          by the gpu process

        */
        TIME_START;
        N_gpu = particle_partition(N, argc, argv);
        N_cpu = N - N_gpu;
        /* observation for the current time step */
        getObs(ob, t);
        if (N_gpu > 0){
            *((int*)addr_n) = N_gpu;
            /* Adaptation is done, now it is time to add 2 to the semaphore */
            semop0(2);
            LOG1("Adaptation done (sema value = %d)", GETSEM);
            /*
              Note that when N_gpu = 0, the semaphore value will not be
              incremented, which means the gpu process will be blocked
              until being killed by the main process.
             */
        }
        if (N_cpu > 0){
            LOG2("CPU starts to work on particles %d ~ %d", N_gpu, N-1);
            /**********************************
             Sampling and importance modules
            ***********************************/
            for (int p = N_gpu; p < N; p ++){
                /* Sampling */
                mdl_propagate(&particles[p*STATE_DIM], p_new);
                /* Importance */
                w = mdl_weight(p_new, ob);
                /* Update memory */
                memcpy(&particles[p*STATE_DIM], p_new, p_sz);
                weights[p] = w;
            }
            LOG1("CPU finished in %lf seconds", ELAPSEDTIME);
        } else
            LOG("Nothing to be done on CPU");

        /*
          Now let's wait for the gpuProcess to finish,
          i.e. when the semaphore becomes zero again
          This happens only when there is some work on the GPU.
        */
        if (N_gpu > 0)
            semop0(0);
        TIME_END;
        time_inc += ELAPSEDTIME;
        LOG1("GPU finished (sema value = %d)", GETSEM);
        /*******************************************
          Resampling
          Systematic global resampling
        ********************************************/
        LOG2("Particle with max weight: (ob: %lf  %lf)", ob[obx], ob[oby]);
        printMax(particles, weights, 0, N);

        ess = prescan(weights, N);
        LOG2("ESS = %lf (N = %d)", ess, N);
        if (ess < N/2) LOG("Resampling NEEDED!!!");

        LOG1(">>>>>>>>>> Iteration %d ends: <<<<<<<<<<", t);
        t++;
    } while (t < t_max);

    printf("%d  %d  %d  %lf\n", N, N_gpu, t_max-1, time_inc / (t_max-1));
    /*
      We are waiting for the gpuProcess to terminate first
      kill the gpu process
     */
    LOG1("Killing the gpu process [%d]", gpuPid);
    if (-1 == kill(gpuPid, SIGTERM)) errExit("kill");
    int status;
    if (-1 == waitpid(gpuPid, &status, WUNTRACED | WCONTINUED))
        errExit("waitpid");
    LOG1("gpu[%d] has terminated", gpuPid);
    /*
      This will unlink the shared memory from the current process
      and the shared memory will be removed from /dev/shm/ when all associated
      processes have unlinked the shared memory.
     */
    shm_unlink("shm_pp");
    shm_unlink("shm_gpu_n");
    shm_unlink("shm_w");
    shm_unlink("shm_ob");
    LOG("Unlinked shared memory");

    /*
      Remove the semaphore set
     */
    if (-1 == semctl(semid, 0, IPC_RMID)) errExit("semctl rmid");
    LOG1("Removed semaphore set %d", semid);
    exit(0);
}

int particle_partition (int n, int argc, char ** argv)
{
    int N_gpu = 1024;
    if (argc > 2){
        N_gpu = getInt(argv[2], 0, "number of particles on GPU");
        if (N_gpu > n)
            errExit("N_gpu is larger than N");
    }
    return N_gpu;
}

void getObs(obs_t *ob, int t)
{
    ob[obx] = (obs_t)(t*dt) + ((obs_t)random()/RAND_MAX-0.5)*.1;
    ob[oby] = (obs_t)(t*dt) + ((obs_t)random()/RAND_MAX-0.5)*.1;
}

void printMax(state_t *particles, double * weight, int start, int end)
{
#ifdef VERBOSE
    double maxw = 0;
    int pid = 0;
    for (int p = start; p < end; p ++){
        if (weight[p] > maxw){
            maxw = weight[p];
            pid = p;
        }
    }
    printf("\t%d: ", pid);
    for (int st = 0; st < STATE_DIM; st ++)
        printf("%lf  ", particles[pid*STATE_DIM+st]);
    printf("(%lf)\n", weight[pid]);
#endif
}

double normalize(double *weights, int N)
{
    double w_sum = 0.0, w2 = 0.0;
    for (int i = 0; i < N; i++){
        w_sum += weights[i];
    }
    for (int i = 0; i < N; i++){
        weights[i] = weights[i]/w_sum;
        w2 += weights[i] * weights[i];
    }
    return 1.0 / w2; // ESS
}

double prescan(double *weights, int N)
{
    double ess = normalize(weights, N);
    double w_inc = 0.0, tmp, w2 = 0.0;
    for(int i = 0; i < N; i ++){
        tmp = weights[i];
        weights[i] += w_inc;
        w_inc += tmp;
    }
    return ess;
}
