# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:28:10 2022

@author: jakob
"""

import argparse
import sys, os
sys.path.append('../source/')



# define parser arguments

#define arguments passed in the command line
parser = argparse.ArgumentParser(description='Takes num_samples, grad_steps, intermediate steps and stepsize')
parser.add_argument('-num_samples', type=int, help='number of samples drawn from the proposal distribution')
parser.add_argument('-grad_steps', type=int, help='number of gradient steps taken')
parser.add_argument('-stepsize', type=float, help='stepsize of the adam optimizer')

args = parser.parse_args()
if args.num_samples:
    num_samples = args.num_samples
else:
    num_samples = 100000
    
if args.grad_steps:
    grad_steps = args.grad_steps 
else:
    grad_steps = 100
    
if args.stepsize:
    stepsize = args.stepsize
else:
    stepsize = 0.005
    

import jax
import jax.numpy as jnp
import jax.random as random

import jax_cosmo as jc

import numpyro
from numpyro import sample
import numpyro.distributions as dist
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer import init_to_value, Trace_ELBO, SVI
from numpyro.infer import MCMC, NUTS
from optax import adam

import matplotlib.pyplot as plt

import arviz as az

# create random keys
key = random.PRNGKey(3141)
key, subkey, subkey2, subkey3 = random.split(key, 4)

#create random data
shape = (2000,)
z_random = random.uniform(key, shape, minval=0.01, maxval=1.5)
z = jnp.sort(z_random)


#true parameter
true_param = {'h':0.7,
    'Omega_m':0.3,
    'w0':-1.0,
    'wa':0.0}

#define distance modulus function
def distance_modulus(theta, z):
    a = jc.utils.z2a(z)
    h = theta.get('h', 0.7)
    Omega_m = theta.get('Omega_m', 0.3)
    Omega_b = 0.05
    Omega_c = Omega_m - Omega_b
    Omega_k = theta.get('Omega_k', 0.)
    w0 = theta.get('w0', -1.)
    wa = theta.get('wa', 0.)
    
    cosmology = jc.Cosmology(h=h, Omega_c=Omega_c, Omega_b=Omega_b, w0=w0, wa=wa, Omega_k=Omega_k, n_s=0.96, sigma8=0.83)
    dist_L = (jc.background.angular_diameter_distance(cosmology, a)/a**2.0)/h
    dist_mod = 25. + 5. * jnp.log10(dist_L)
    return dist_mod


#distance moduli with/without gaussian error

sigma = 0.3 * jnp.log(1+z)
dist_mod_err = sigma * random.normal(subkey, shape) 

cov = jnp.diag(sigma**2)
inv_cov = jnp.linalg.inv(cov)

dist_mod = distance_modulus(true_param, z) #+ dist_mod_err


#setting up the model
def model_cpl(redshift, distance_mod, covariance):
    Omega_m = sample("Omega_m", dist.Uniform(0,1))
    w0 = sample("w0", dist.Uniform(-10, 10))
    wa = sample("wa", dist.Uniform(-10,10))
    theta_dict = {"Omega_m":Omega_m, "w0":w0, "wa":wa}
    mu = distance_modulus(theta_dict, redshift)
    
    sample("y", dist.MultivariateNormal(mu, covariance), obs=distance_mod)
    
initial_values = {'Omega_m':0.3, 'w0':-1., 'wa':0.}

guide_cpl = AutoMultivariateNormal(model_cpl, 
                                   init_loc_fn = init_to_value(values=initial_values))

svi = SVI(model_cpl, 
    guide_cpl, 
    adam(stepsize), 
    Trace_ELBO(),
    redshift=z, 
    distance_mod=dist_mod, 
    covariance = cov
)

svi_result = svi.run(subkey2, grad_steps)

samples = guide_cpl.sample_posterior(subkey3, svi_result.params, (num_samples,))
idata_num = az.from_dict(samples)

idata_num.to_netcdf('jax_cosmo_results/adiv_res.nc')