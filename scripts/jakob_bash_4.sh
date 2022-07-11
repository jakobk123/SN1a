#!/bin/bash


echo "running emcee"
START1="$(date +%s)"
python emcee_jax_cosmo.py -num_steps 20000 -nwalkers 16

echo "The emcee part of the code took: $[ $(date +%s) - ${START1} ]"

echo "running ADVI"
START2="$(date +%s)"
python advi_jax_cosmo.py -num_samples 100000 -grad_steps 25000 -stepsize 0.005

echo "The ADVI part of the code took: $[ $(date +%s) - ${START2} ]"

echo "running NUTS"
START3="$(date +%s)"
python nuts_jax_cosmo.py -num_samples 20000 -burnin 500 -nwalkers 16

echo "The NUTS part of the code took: $[ $(date +%s) - ${START3} ]"


echo "runs finished"
