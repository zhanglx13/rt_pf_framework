#ifndef __MODEL_VAR__
#define __MODEL_VAR__

enum state
{
    Delta,
    G,
    Xram,
    Me,
    mu,
    phe,
    Scl,
    Smr,
    mI,
    STATE_DIM
};

enum observation
{
    ob_I,    // y2
    ob_Me,   // y3
    ob_Xram, // y1
    ob_G,    // y0
    ob_Smr,  // y5
    ob_Scl,  // y4
    ob_phe,  // y6
    OB_DIM
};

enum input
{
    in_Ic,   // u0
    in_Vramc,// u1
    INPUT_DIM
};

typedef double state_t;
typedef double ob_t;
typedef double input_t;

#define pi  3.1415926
#define De  43.18 //Electrode diameter (in)
#define Di  50.8 //Ingot diameter (in)
#define I0  6000.0 //Nominal current (A)
#define G0  0.9144 //Nominal gap (in)
#define phe0  3.0 //Nominal helium pressure (Torr)
#define mu0  0.440087 //Nominal melting efficiency
#define Vc  21.18 //Cathode voltage fall (V)
#define Ri  4.37e-4 //Gap-independent electric resistance (Ohm)
#define Rg  0.0 //Gap-dependent electric resistance (Ohm/cm)
#define Scl0  15.702565 //Nominal centerline pool depth (cm)
#define Smr0  13.234694 //Nominal mid-radius pool depth (cm)
#define Acl  1.909e-3 //A matrix (1)
#define Amr  1.443e-3 //
#define Bdeltacl  2.60e-5 //Bdelta matrix (1)
#define Bdeltamr  (-1.29e-4) //
#define Bicl  6.587e-6 //Bi matrix (cm/A)
#define Bimr  3.165e-6 //
#define Bmucl  6.899e-2 //Bmu matrix (cm)
#define Bmumr  2.686e-2 //
#define Bhecl  (-8.091e-4) //Bhe matrix (cm/Torr)
#define Bhemr  (-6.541e-4)
#define sigmaI  20.0 //Current standard deviation (A)
#define sigmaVram  5.0e-4 //Ram speed standard deviation (cm/s)
#define sigmamur  0.0044 //Melting efficiency standard deviation (1)
#define sigmaa  2.7750e-4 //Fill ratio standard deviation (1)
#define sigmaVb  0.0238020 //Voltage bias standard deviation
#define sigmaIb  6.0 //Current bias standard deviation
#define sigmaVramb  1.47944e-06 //Ram speed bias standard deviation
#define sigmahe  0.003 //Helium pressure standard deviation
#define sigmaG  0.2 //Measured electrode gap standard deviation (cm)
#define sigmaPos  0.005 //Measured ram position standard deviation (cm)
#define sigmaImeas  15.0 //Measured current standard deviation (A)
#define sigmaLC  200.0 //Measured load cell standard deviation (g)
#define sigmaVmeas  0.1 //Measured voltage standard deviation (V)
#define sigmaCL  1.0 //Measured centerline pool depth standard deviation (cm)
#define sigmaMR  1.0 //Measured mid-radius pool depth standard deviation (cm)
#define sigmahemeas  1.0e-2 //Measured helium pressure standard deviation (Torr)
#define dt  6.0 //Time step (s)
// Other global variables
#define alphar  0.023821
#define alpham  0.059553
#define hr  0.0
#define rhor  7.75
#define hm  5.4153125e+3
#define hs  8.0503125e+3
#define a0  0.2775
#define betam  1.5
#define Cdd  39.86834491
#define Cdp  3.83122724
#define Csd  6.70469459
#define Csp  1.316984365
#define delta0  31.2770456
#define Vram0  0.00148
#define epsilon  1.0e-10
#define var_delta0  5.0
#define G11  (-5.61818883e-06)
#define G21  (5.35922544e-07)
#define G41  (-0.02191774646)

void mdl_init_particles(state_t *particles, unsigned int N);
void mdl_propagate(state_t *p_old, state_t *p_new, input_t * input);
double mdl_weight(state_t *p, ob_t *ob);
void getObs(ob_t *obs, int t);
void getInputs(input_t *inputs, int t);

#endif
