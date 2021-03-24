#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "model_VAR.h"

#define NDATA 1202
ob_t obs_full[NDATA][OB_DIM];
input_t inputs_full[NDATA][INPUT_DIM];

/*
  generate 2 independent random numbers from normal distribution
  Copied from https://rosettacode.org/wiki/Statistics/Normal_distribution
 */
void gen_rand_normal(double *r1, double *r2)
{
    double x,y,rsq,f;
    do {
        x = 2.0 * random() / (double)RAND_MAX - 1.0;
        y = 2.0 * random() / (double)RAND_MAX - 1.0;
        rsq = x * x + y * y;
    }while( rsq >= 1. || rsq == 0. );
    f = sqrt( -2.0 * log(rsq) / rsq );
    *r1 = x * f;
    *r2 = y * f;
}

void load_obs(const char *obs_filename)
{
    FILE *fObs = fopen(obs_filename, "rt");
    char line[512];
    int l = 0;
    while (fgets(line, 512, fObs) != NULL){
        sscanf(line, "%lf  %lf  %lf  %lf  %lf  %lf  %lf\n",
               &obs_full[l][ob_I],
               &obs_full[l][ob_Me],
               &obs_full[l][ob_Xram],
               &obs_full[l][ob_G],
               &obs_full[l][ob_Smr],
               &obs_full[l][ob_Scl],
               &obs_full[l][ob_phe]
            );
        l++;
    }
    fclose(fObs);
}

void load_inputs(const char *inputs_filename)
{
    FILE *fIn = fopen(inputs_filename, "rt");
    char line[100];
    int l = 0;
    while (fgets(line, 100, fIn) != NULL){
        sscanf(line, "%lf  %lf\n",
               &inputs_full[l][in_Ic],
               &inputs_full[l][in_Vramc]
            );
        l++;
    }
    fclose(fIn);
}



void mdl_init_particles(state_t *particles,  unsigned int N)
{
    srandom(time(NULL));
    ob_t ob[OB_DIM];
    input_t input[INPUT_DIM];
    load_obs("VAR_data/data_obs.txt");
    load_inputs("VAR_data/data_inputs.txt");
    /* test obs reading */
    /* for (int l = 0; l < 1202; l++){ */
    /*     printf("%7.2f  %9.1f  %11.8f  %8.6f  %9.6f  %9.6f  %8.6f\n", */
    /*            obs_full[l][ob_I], */
    /*            obs_full[l][ob_Me], */
    /*            obs_full[l][ob_Xram], */
    /*            obs_full[l][ob_G], */
    /*            obs_full[l][ob_Smr], */
    /*            obs_full[l][ob_Scl], */
    /*            obs_full[l][ob_phe] */
    /*         ); */
    /* } */

    /* test input reading */
    /* for (int l = 0; l < 1202; l++){ */
    /*     printf("%7.2f  %10.8f\n", */
    /*            inputs_full[l][Ic], */
    /*            inputs_full[l][Vramc] */
    /*         ); */
    /* } */
    getObs(ob, 0);
    getInputs(input, 0);

    double Sdot_i = 4.0 * mu0 * (Vc + (Ri + Rg * ob[ob_G]) * input[in_Ic])
        * input[in_Ic] / hs / pi / De / De;
    double delta_i = 7.0 * alphar * (0.5 + betam / 3.0) / Sdot_i;
    double x0,x1,x2,x3,x4,x5,x6,x7,x8;		//9 states of one particle
    double rnn0, rnn1;
    for (unsigned int p = 0; p < N; p ++){
        gen_rand_normal(&rnn0, &rnn1);
        x0 = delta_i + sqrt(var_delta0) * rnn0;
	x1 = ob[ob_G] + sigmaG * rnn1;
        gen_rand_normal(&rnn0, &rnn1);
        x2 = ob[ob_Xram] + sigmaPos * rnn0;
	x3 = ob[ob_Me] + sigmaLC * rnn1;
        gen_rand_normal(&rnn0, &rnn1);
        x4 = mu0 + sqrt(dt)*sigmamur * rnn0;
        x5 = ob[ob_phe] + sqrt(dt)*sigmahe * rnn1;
        gen_rand_normal(&rnn0, &rnn1);
	x6 = ob[ob_Scl] + sigmaCL * rnn0;
	x7 = ob[ob_Smr] + sigmaMR * rnn1;
        gen_rand_normal(&rnn0, &rnn1);
	x8 = input[in_Ic] + sigmaImeas * rnn0;
        particles[p * STATE_DIM + 0] = x0;
        particles[p * STATE_DIM + 1] = x1;
        particles[p * STATE_DIM + 2] = x2;
        particles[p * STATE_DIM + 3] = x3;
        particles[p * STATE_DIM + 4] = x4;
        particles[p * STATE_DIM + 5] = x5;
        particles[p * STATE_DIM + 6] = x6;
        particles[p * STATE_DIM + 7] = x7;
        particles[p * STATE_DIM + 8] = x8;
    }
}

void getObs(ob_t *obs, int t)
{
    for (int i = 0; i < OB_DIM; i++)
        obs[i] = obs_full[t][i];
}

void getInputs(input_t *inputs, int t)
{
    for (int i = 0; i < INPUT_DIM; i++)
        inputs[i] = inputs_full[t][i];
}

void mdl_propagate(state_t *p_old, state_t *p_new, input_t *input)
{
    double rnn0, rnn1, mI;
    // Reconstruct inputs
    double  Vram, P;
    Vram = input[in_Vramc];
    P = (Vc + Ri * p_old[8]) * p_old[8];
    /* Sample process uncertainties */
    gen_rand_normal(&rnn0,&rnn1);
    double dI = dt * sigmaI * rnn0;
    double dVram = dt * sigmaVram * rnn1;
    gen_rand_normal(&rnn0,&rnn1);
    double dmu = sqrt(dt) * sigmamur * rnn0;
    double dhe = sqrt(dt) * sigmahe * rnn1;
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
    /* Discrete propagation equations */
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
    mI = input[in_Ic] + (input[in_Ic] - p_old[8])*exp(-dt/1.0);
    p_new[8] = mI + dI;
}

double mdl_weight(state_t *p, ob_t *ob)
{
    double w = 0.0;
    double y_G = p[1];
    double y_Xram = p[2];
    double y_I = p[8];
    double y_Me = p[3];
    double y_Scl = p[6];
    double y_Smr = p[7];
    double y_phe = p[5];
    double expo =
        (ob[ob_G]    - y_G)    * (ob[ob_G]    - y_G)    / sigmaG/sigmaG +
        (ob[ob_Xram] - y_Xram) * (ob[ob_Xram] - y_Xram) / sigmaPos/sigmaPos +
        (ob[ob_I]    - y_I)    * (ob[ob_I]    - y_I)    / sigmaImeas/sigmaImeas +
        (ob[ob_Me]   - y_Me)   * (ob[ob_Me]   - y_Me)   / sigmaLC/sigmaLC +
        (ob[ob_Scl]  - y_Scl)  * (ob[ob_Scl]  - y_Scl)  / sigmaCL/sigmaCL +
        (ob[ob_Smr]  - y_Smr)  * (ob[ob_Smr]  - y_Smr)  / sigmaMR/sigmaMR +
        (ob[ob_phe]  - y_phe)  * (ob[ob_phe]  - y_phe)  / sigmahemeas/sigmahemeas;
    w = exp(-0.5 * expo);
    return w+1e-99;
}
