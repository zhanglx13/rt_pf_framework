#ifndef __MODEL_TRIV__
#define __MODEL_TRIV__


enum state
{
    stx,
    sty,
    STATE_DIM
};

enum observation
{
    obx,
    oby,
    OBS_DIM
};

typedef double state_t;
typedef double obs_t;

#define dt 0.1

void mdl_init_particles(state_t *particles, unsigned int n);
void mdl_propagate(state_t *p_old, state_t *p_new);
double mdl_weight(state_t *p, obs_t *ob);
/*
  get observations for the current time step
*/
void getObs(obs_t *ob, int t);

#endif
