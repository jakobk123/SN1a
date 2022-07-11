# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:41:17 2022

@author: jakob
"""


import argparse
import sys, os
sys.path.append('../source/')



# define parser arguments

#define arguments passed in the command line
parser = argparse.ArgumentParser(description='Takes num_steps and burnin and nwalkers')
parser.add_argument('-num_samples', type=int, help='number of samples drawn from the proposal distribution')
parser.add_argument('-burnin', type=int, help='burnin')
parser.add_argument('-nwalkers', type=int, help='number of walkers')


args = parser.parse_args()
if args.num_samples:
    num_samples = args.num_samples
else:
    num_samples = 100
    
if args.burnin:
    burnin = args.burnin 
else:
    burnin = 100
    
if args.nwalkers:
    nwalkers = args.nwalkers 
else:
    nwalkers = 8
    

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
    

kernel = NUTS(model_cpl, 
              step_size=1e-1, 
              init_strategy=numpyro.infer.init_to_median, 
              dense_mass=True, 
              max_tree_depth=5
             )
mcmc = MCMC(kernel, 
            num_warmup=burnin, 
            num_samples=num_samples, 
            num_chains=nwalkers, 
            chain_method='vectorized'
           )
mcmc.run(subkey2, redshift=z, 
    distance_mod=dist_mod, 
    covariance = cov)

idata = az.from_numpyro(mcmc)
az.plot_trace(idata);

idata.to_netcdf('jax_cosmo_results/nuts_res.nc')