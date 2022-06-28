#!/bin/bash


samps=75000

echo "running numpyro"
START2="$(date +%s)"
python numpyro_pantheon.py -num_samples 100000 -grad_steps $samps -intermediate_steps 20 -stepsize 0.001

echo "The numpyro part of the code took: $[ $(date +%s) - ${START2} ]"



echo "runs finished"
