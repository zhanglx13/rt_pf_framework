# Real-time Framework for Particle Filter Applications


1. To compile
  - If want to see LOG info, do
    make MODEL=triv verbose=1
  - Else, do
    make MODEL=triv
2. To run
  ./main N N_gpu t_max
  where N is the total number of particles, default: 5000
  N_gpu is the number of particles assigned to GPU, default: 1024
  t_max is the number of iterations + 1, default: 101
  If you want to specify the ith argument, you need to specify all arguments
  that comes before the ith argument.

## Model --- triv

## Model --- VAR

## Model --- EKD
