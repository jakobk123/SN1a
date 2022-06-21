#!/bin/bash

echo "running emcee"
START1="$(date +%s)"
python emcee_pantheon.py -num_samples 100000 -nwalkers 16

echo "The emcee part of the code took: $[ $(date +%s) - ${START1} ]"

echo "running numpyro"
START2="$(date +%s)"
python numpyro_pantheon.py -num_samples 100000 -grad_steps 50000 -intermediate_steps 20 -stepsize 0.001

echo "The numpyro part of the code took: $[ $(date +%s) - ${START2} ]"


echo "runs finished"
