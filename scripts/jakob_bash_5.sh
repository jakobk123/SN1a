#!/bin/bash



echo "running ADVI"
START2="$(date +%s)"
python advi_jax_cosmo.py -num_samples 100000 -grad_steps 25000 -stepsize 0.005

echo "The ADVI part of the code took: $[ $(date +%s) - ${START2} ]"

echo "running NUTS"
START3="$(date +%s)"
python nuts_jax_cosmo.py -num_samples 25000 -burnin 500 -nwalkers 16

echo "The NUTS part of the code took: $[ $(date +%s) - ${START3} ]"

rune=$1

if [[ "$rune" == "emcee" ]]
then
echo "running emcee"
START1="$(date +%s)"
python emcee_jax_cosmo.py -num_steps 25000 -nwalkers 16
echo "The emcee part of the code took: $[ $(date +%s) - ${START1} ]"
fi


echo "runs finished"
