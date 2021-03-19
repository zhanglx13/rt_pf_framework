#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "model_triv.h"


void mdl_init_particles(state_t *particles, unsigned int n)
{
    srandom(time(NULL));
    for (unsigned int p = 0; p < n; p++){
        for (int st = 0; st < STATE_DIM; st ++){
            double u = ((double)random()/RAND_MAX)*10.0;
            particles[p * STATE_DIM + st] = (state_t)u;
        }
    }
}

void mdl_propagate(state_t *p_old, state_t *p_new)
{
    // x = x + 1 + u; --> particles[p*STATE_DIM + 0]
    // y = y + 1 + v; --> particles[p*STATE_DIM + 1]
    double ux = (double)random()/RAND_MAX - 0.5;
    double uy = (double)random()/RAND_MAX - 0.5;
    p_new[0] = p_old[0] + 1.0*dt + (state_t)ux*dt;
    p_new[1] = p_old[1] + 1.0*dt + (state_t)uy*dt;
}

/*
  The weight of each particle is set to be the reciprocal of (distance+1)
  Since distance can be zero, +1 will limit the maximum weight to be 1.
 */
double mdl_weight(state_t *p, obs_t *ob)
{
    double dis = sqrt((p[0]-ob[0])*(p[0]-ob[0]) + (p[1]-ob[1])*(p[1]-ob[1]));
    return 1.0 / (dis+1.0);
}
