#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 19:08:45 2022

@author: jakob
"""


import numpy as np
import argparse
import sys, os
sys.path.append('../source/')
from cosmo_jnp import cosmo, pantheon
from run import run_sampler




#import data. The d denotes the union 2.1 data, p pantheon data and tot is the combined data. 
#inv stands for the inverse covariance matrix


data_dir = '../data/'
p = pantheon(data_dir+'pantheon.txt', data_dir+'pantheon_covsys.txt')


panth_dict = {'x' : p.redshift1, 'y' : p.dm, 'inv_cov' : p.inv_cov, 'name':'panth'}

#defining cosmology object
cosmology = cosmo(0.7, p.redshift1)


#defining log_likelihood

def log_likelihood(y, inv_cov, cosm, theta_dict):
    distance_mod = cosm.dist_mod(theta_dict)
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
    return lp + log_likelihood( y, inv_cov, cosm,theta_dict)

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
    return lp + log_likelihood( y, inv_cov, cosm,theta_dict)

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
    return lp + log_likelihood( y, inv_cov, cosm,theta_dict)

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





results_dir='emcee_results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
    
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
    nwalkers = 16
    



for i in [lcdm_dict, cpl_dict, alpha_dict]:
    for j in joint_dict.keys():
        try:
            idata_0 = run_sampler(i, 'init0', nwalkers, panth_dict['y'], panth_dict['inv_cov'], cosmology, num_steps, i['name'], joint_dict, j, results_dir=results_dir)
        except Exception as err:
            print('Exception raised! {}'.format(err))
            
        try:
            idata_1 = run_sampler(i, 'init1', nwalkers, panth_dict['y'], panth_dict['inv_cov'], cosmology, num_steps, i['name'], joint_dict, j, results_dir=results_dir)
        except Exception as err:
            print('Exception raised! {}'.format(err))
            

