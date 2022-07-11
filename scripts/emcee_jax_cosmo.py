# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 14:16:28 2022

@author: jakob
"""

import numpy as np
import argparse
import sys, os
sys.path.append('../source/')



# define parser arguments

#define arguments passed in the command line
parser = argparse.ArgumentParser(description='Takes number of steps and number of walkers')
parser.add_argument('-num_steps', type=int, help='number of steps in for one walker')
parser.add_argument('-nwalkers', type=int, help='number of walkers')

args = parser.parse_args()
if args.num_steps:
    num_steps = args.num_steps
else:
    num_steps = 100
    
if args.nwalkers:
    nwalkers = args.nwalkers 
else:
    nwalkers = 8
    

import jax
import jax.numpy as jnp
import jax.random as random

import jax_cosmo as jc

import matplotlib.pyplot as plt

import arviz as az

import emcee as em

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

from scipy.constants import c

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
    
    #cosmology = jc.Cosmology(h=h, Omega_c=Omega_c, Omega_b=Omega_b, w0=w0, wa=wa, Omega_k=Omega_k, n_s=0.96, sigma8=0.83)
    
    H_0 = 100 * h * 1000 / c
    z_arr = np.linspace(0., z, 500)
    H_z = (H_0 * np.reciprocal(np.exp(np.log(1+z_arr)))) * np.sqrt(Omega_m * (1+z_arr) ** 3 + 
                                                   (1 - Omega_m)*(1+z_arr)**(3*(1+w0+wa))*np.exp(-3*wa/(1+z_arr)))
    d_L = (1+z) * np.trapz(1/H_z, x=z_arr, axis=0)
    
    #dist_L = (jc.background.angular_diameter_distance(cosmology, a)/a**2.0)/h
    #dist_mod = 25. + 5. * jnp.log10(dist_L)
    dist_mod = 25 + 5 * np.log10(d_L)
    return dist_mod


#distance moduli with/without gaussian error

sigma = 0.3 * jnp.log(1+z)
dist_mod_err = sigma * random.normal(subkey, shape) 

cov = jnp.diag(sigma**2)
inv_cov = jnp.linalg.inv(cov)

dist_mod = distance_modulus(true_param, z) #+ dist_mod_err



#define probability functions
def log_likelihood(theta_dict, x, y, inv_cov):
    distance_mod = np.asarray(distance_modulus(theta_dict, x) , dtype=float)
    log_like = -0.5 * ((y - distance_mod) @ inv_cov @ (y - distance_mod))
    return log_like

def log_prior(Omega_m=0.3, w_0=-1, w_1=0):
    '''returns the log of the propability densities up to a constant'''
    if 0.0 < Omega_m < 1.0:
        return -((w_0+1.0)**2.0/5.0**2.0 + (w_1)**2.0/5.0**2.0) 
    return -np.inf

def log_probability(theta, x, y, inv_cov):
    '''performs the multiplication prior*likelihood'''
    keys = ['Omega_m', 'w_0', 'w_1']
    theta_dict = dict(zip(keys, theta))
    lp = log_prior(**theta_dict)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta_dict, x, y, inv_cov)



#print(name+' started...')
initial = [0.3, -1, 0.01]
ndim = 3
pos = initial + 0.05 * np.random.randn(nwalkers, ndim) 

sampler = em.EnsembleSampler(
nwalkers, ndim, log_probability, args=(z,dist_mod,inv_cov)#, backend=backend
)
sampler.run_mcmc(pos, num_steps, progress=True)
samples = sampler.get_chain(flat=True) 

var_names = ['Omega_m', 'w0', 'wa']
idata_em = az.from_emcee(sampler, var_names=var_names)

idata_em.to_netcdf('jax_cosmo_results/emcee_res.nc')