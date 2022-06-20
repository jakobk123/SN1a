#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 19:08:45 2022

@author: jakob
"""

import emcee as em
import arviz as az
import numpy as np

from cosmo_jnp import cosmo, pantheon

import jax.numpy as jnp



#import data. The d denotes the union 2.1 data, p pantheon data and tot is the combined data. 
#inv stands for the inverse covariance matrix


p = pantheon('pantheon.txt', 'pantheon_covsys.txt')
p_inv = p.inv_cov
p_cov = jnp.linalg.inv(p_inv)

panth_dict = {'x' : p.redshift1, 'y' : p.dm, 'cov':p_cov, 'inv_cov' : p_inv, 'name':'panth'}

#defining cosmology object
cosmology = cosmo(0.7, p.redshift1)


#defining log_likelihood

def log_likelihood(y, inv_cov, cosm, Omega_m=0.3, w_0=-1, w_1=0, a=0,):
    distance_mod = cosm.dist_mod(Omega_m, w_0, w_1, a)
    log_like = -0.5 * ((y - distance_mod) @ inv_cov @ (y - distance_mod))
    #log_like = - 0.5 * np.sum(((y-distance_mod)/cov.err)**2)
    return log_like

#defining priors

def log_prior(Omega_m=0.3, w_0=-1, w_1=0, a=0):
    '''returns the log of the propability densities up to a constant'''
    if 0.0 < Omega_m < 1:
        return -((w_0+1)**2/5**2 + (w_1)**2/5**2 + a**2/5**2) 
    return -np.inf

def log_prior_planck(Omega_m, w_0=-1, w_1=0, a=0):
    '''returns the log of the propability densities up to a constant'''
    l_prior = - ((Omega_m - 0.3166)**2/0.0084**2 + (w_0+0.957)**2/0.08**2 + (w_1 + 0.32)**2/0.29**2 + a**2/0.5**2) 
    return l_prior

def log_no_prior(Omega_m=0.3, w_0=-1, w_1=0, a=0):
    '''returns the log of the propability densities up to a constant'''
    if 0.0 < Omega_m < 1:
        return 0
    return -np.inf

#defining log_probability

def log_probability(theta, y, inv_cov, cosm):
    '''
    performs the multiplication prior*likelihood

    Parameters
    ----------
    theta : list of parameter values. Float
    y : distance modulus data
    inv_cov : inverse covariance matrix
    cosm : cosmology type cosmo_class

    Returns
    -------
    Log of joint probability
    Device array

    '''
    
    keys = ['Omega_m', 'w_0', 'w_1', 'a']
    theta_dict = dict(zip(keys, theta))
    lp = log_prior(**theta_dict)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood( y, inv_cov, cosm,**theta_dict)

def log_probability_planck(theta, y, inv_cov, cosm):
    '''
    performs the multiplication prior*likelihood

    Parameters
    ----------
    theta : list of parameter values. Float
    y : distance modulus data
    inv_cov : inverse covariance matrix
    cosm : cosmology type cosmo_class

    Returns
    -------
    Log of joint probability
    Device array

    '''
    
    keys = ['Omega_m', 'w_0', 'w_1', 'a']
    theta_dict = dict(zip(keys, theta))
    lp = log_prior_planck(**theta_dict)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood( y, inv_cov, cosm,**theta_dict)

def log_probability_noprior(theta, y, inv_cov, cosm):
    '''
    performs the multiplication prior*likelihood

    Parameters
    ----------
    theta : list of parameter values. Float
    y : distance modulus data
    inv_cov : inverse covariance matrix
    cosm : cosmology type cosmo_class

    Returns
    -------
    Log of joint probability
    Device array

    '''
    
    keys = ['Omega_m', 'w_0', 'w_1', 'a']
    theta_dict = dict(zip(keys, theta))
    lp = log_no_prior(**theta_dict)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood( y, inv_cov, cosm,**theta_dict)

#setting initial values for emcee
initial_lambda = np.array([0.5])
initial_lambda2 = np.array([0.3])
initial_cpl = np.array([0.5, 0, 1])
initial_cpl2 = np.array([0.3, -1, 0])
initial_alpha = np.array([0.5, 0, 1, 1])
initial_alpha2 = np.array([0.3, -1, 0, 0])

lcdm_dict = {'init0' : initial_lambda, 'init1': initial_lambda2, 'name' : 'lcdm'}
cpl_dict = {'init0' : initial_cpl, 'init1': initial_cpl2, 'name' : 'cpl'}
alpha_dict = {'init0' : initial_alpha, 'init1': initial_alpha2, 'name' : 'alpha'}

joint_dict = {'prior0': log_probability, 'prior1': log_probability_planck, 'prior2': log_probability_noprior}



#setting run_sampler method
def run_sampler(dictionary, key, nwalkers,  dm, data_inv, cosm, num, name, prob_dict, prob_key):
    
    initial = dictionary[key]
    log_probability = joint_dict[prob_key]
    
    print(name+' started...')

    ndim = len(initial)
    pos = initial + 0.05 * np.random.randn(nwalkers, ndim) 
    

    #filename = name+"_backend.h5"
    #backend = em.backends.HDFBackend(filename)
    #backend.reset(nwalkers, ndim)
    
    sampler = em.EnsembleSampler(
    nwalkers, ndim, log_probability, args=(dm, data_inv, cosm)#, backend=backend
    )
    sampler.run_mcmc(pos, num, progress=True);
    
    
    #saving the data as a numpy-readable file is optional
    #np.savetxt('data_tables/'+name+'.gz', samples)
    
    labels = ['Omega_m', 'w_0', 'w_1', 'a']
    var_names = labels[slice(None, ndim)]
    
    idata = az.from_emcee(sampler, var_names=var_names)
    
    filename = name+'_'+key+'_'+prob_key
    
    print(filename)
    
    file = idata.to_netcdf('emcee_results/'+filename+'.nc', groups='posterior')
    
    return idata


nwalkers = 32
num_steps = 100

for i in [lcdm_dict, cpl_dict, alpha_dict]:
    for j in joint_dict.keys():
        try:
            idata_0, file = run_sampler(i, 'init0', nwalkers, panth_dict['y'], panth_dict['inv_cov'], cosmology, num_steps, i['name'], joint_dict, j)
        except ValueError as err:
            print('Exception raised! {}'.format(err))
            
        try:
            idata_1 = run_sampler(i, 'init1', nwalkers, panth_dict['y'], panth_dict['inv_cov'], cosmology, num_steps, i['name'], joint_dict, j)
        except Exception:
            print('Exception raised! ')
            

