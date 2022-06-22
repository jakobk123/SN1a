#!/bin/bash


samps=25000

echo "running emcee"
START1="$(date +%s)"
python emcee_pantheon.py -num_steps $samps -nwalkers 16

echo "The emcee part of the code took: $[ $(date +%s) - ${START1} ]"

echo "running numpyro"
START2="$(date +%s)"
python numpyro_pantheon.py -num_samples 100000 -grad_steps $samps -intermediate_steps 20 -stepsize 0.001

echo "The numpyro part of the code took: $[ $(date +%s) - ${START2} ]"


echo "runs finished"
