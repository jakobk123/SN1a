#!/bin/bash


samps=75000

echo "running emcee"
START1="$(date +%s)"
python emcee_pantheon.py -num_steps $samps -nwalkers 16

echo "The emcee part of the code took: $[ $(date +%s) - ${START1} ]"



echo "runs finished"
