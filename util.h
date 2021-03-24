#ifndef __UTIL_H__
#define __UTIL_H__

/*
  POSIX.1-2008 marks gettimeofday() as obsolete,
  recommending the use of clock_gettime(2) instead
  (P491-TLPI)
*/
#define TIME_START clock_gettime(CLOCK_REALTIME, &tp_start);
#define TIME_END   clock_gettime(CLOCK_REALTIME, &tp_end);
#define ELAPSEDTIME                                     \
    ((tp_end.tv_nsec - tp_start.tv_nsec)/1000000000.0 + \
     (tp_end.tv_sec - tp_start.tv_sec))

/*
  Note that the O_EXCL flag is not recommended at this stage of development.
  This is because the program may terminates at any time before we have a chance
  to close these shared memory objects.
 */
#define create_shm(fd, name, size, addr)                                \
    fd = shm_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);           \
    if (-1 == fd) errExit("shm_open" name);                             \
    if (ftruncate(fd, size) == -1) errExit("ftruncate" name);           \
    addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); \
    if (addr == MAP_FAILED) errExit("mmap" name);

/*
  The O_CREAT flag is not specified because we assume that the main process has
  already created the shared memory objects.
*/
#define open_shm(fd, name, size, addr)                                  \
    fd = shm_open(name, O_RDWR, S_IRUSR | S_IWUSR);                     \
    if (-1 == fd) errExit("shm_open" name);                             \
    if (ftruncate(fd, size) == -1) errExit("ftruncate" name);           \
    addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0); \
    if (addr == MAP_FAILED) errExit("mmap" name);

/*
  macro to do semop on the 0th semaphore in semid
 */
#define semop0(val)  sops.sem_num = 0; sops.sem_op = val; sops.sem_flg = 0; \
    if (-1 == semop(semid, &sops, 1)) errExit("semop");
#define GETSEM semctl(semid, 0, GETVAL)

/*
  Logging macros taking different arguments
 */
#ifdef VERBOSE
#define LOG(msg)                 \
    printf("%s %s[%d]: " msg "\n", currTime("%T"), argv[0], getpid());
#define LOG1(msg, value)         \
    printf("%s %s[%d]: " msg "\n", currTime("%T"), argv[0], getpid(), value);
#define LOG2(msg, val1, val2)    \
    printf("%s %s[%d]: " msg "\n", currTime("%T"), argv[0], getpid(), val1, val2);
#define printp(start, end)                                      \
    for (int p = start; p < end; p++){                          \
        printf("\t%*d: ", width, p);                            \
        for (int st = 0; st < STATE_DIM; st ++)                 \
            printf("%9.6lf  ", particles[p * STATE_DIM + st]);  \
        printf("\t(%lf)\n", weights[p]);}
#define printOb(ob)                   \
    printf("\tObs: ");                \
    for (int o = 0; o < OB_DIM; o ++) \
        printf("%lf  ", ob[o]);       \
    printf("\n");
#else
#define LOG(msg)
#define LOG1(msg, val)
#define LOG2(msg, val1, val2)
#define printp(start, end)
#define printOb(ob)
#endif

#endif
