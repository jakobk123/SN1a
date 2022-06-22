# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:15:20 2022

@author: jakob
"""


import sys, os
import argparse

sys.path.append('../source')


#define arguments passed in the command line
parser = argparse.ArgumentParser(description='Takes num_samples, grad_steps, intermediate steps and stepsize')
parser.add_argument('-num_samples', type=int, help='number of samples drawn from the proposal distribution')
parser.add_argument('-grad_steps', type=int, help='number of gradient steps taken')
parser.add_argument('-intermediate_steps', type=int, help='number of intermediate steps between passed to jnp.geomspace')
parser.add_argument('-stepsize', type=float, help='stepsize of the adam optimizer')

args = parser.parse_args()
if args.num_samples:
    num_samples = args.num_samples
else:
    num_samples = 1000
    
if args.grad_steps:
    grad_steps = args.grad_steps 
else:
    grad_steps = 100
    
if args.intermediate_steps:
    intermediate_steps = args.intermediate_steps 
else:
    intermediate_steps = 2
    
if args.stepsize:
    stepsize = args.stepsize
else:
    stepsize = 0.001

from cosmo_jnp import cosmo, pantheon
from run import run_svi_manual

import jax.numpy as jnp




import numpyro.distributions as dist
from numpyro import sample


# load the data
data_dir = '../data/'
p = pantheon(data_dir+'pantheon.txt', data_dir+'pantheon_covsys.txt')
x = p.redshift1
y = p.dm
cov = p.cov


#defining cosmology object
cosmology = cosmo(0.7, p.redshift1)


#defining the models
def model_lcdm(x, y, cov):
    cosm = cosmo(0.7, x)
    Omega_m = sample("Omega_m", dist.Uniform(0,1))
    theta_dict = {"Omega_m":Omega_m}
    mu = cosm.dist_mod(theta_dict)
    
    sample("y", dist.MultivariateNormal(mu, cov), obs=y)
    
    
def model_lcdm_planck(x, y, cov):
    cosm = cosmo(0.7, x)
    Omega_m = sample("Omega_m", dist.TruncatedNormal(0.3166,0.0084, low=0, high=1))
    theta_dict = {"Omega_m":Omega_m}
    mu = cosm.dist_mod(theta_dict)
    
    sample("y", dist.MultivariateNormal(mu, cov), obs=y)

def model_cpl(x, y, cov):
    cosm = cosmo(0.7, x)
    Omega_m = sample("Omega_m", dist.Uniform(0,1))
    w_0 = sample("w_0", dist.TruncatedNormal(-1,5))
    w_1 = sample("w_1", dist.TruncatedNormal(0,5))
    theta_dict = {"Omega_m":Omega_m, "w_0":w_0, "w_1":w_1}
    mu = cosm.dist_mod(theta_dict)
    
    sample("y", dist.MultivariateNormal(mu, cov), obs=y)
    
def model_cpl_planck(x, y, cov):
    cosm = cosmo(0.7, x)
    Omega_m = sample("Omega_m", dist.TruncatedNormal(0.3166,0.0084, low=0, high=1))
    w_0 = sample("w_0", dist.TruncatedNormal(-0.957, 0.08))
    w_1 = sample("w_1", dist.TruncatedNormal(-0.32, 0.29))
    theta_dict = {"Omega_m":Omega_m, "w_0":w_0, "w_1":w_1}
    mu = cosm.dist_mod(theta_dict)
    
    sample("y", dist.MultivariateNormal(mu, cov), obs=y)
    
def model_cpl_noprior(x, y, cov):
    cosm = cosmo(0.7, x)
    Omega_m = sample("Omega_m", dist.Uniform(0,1))
    w_0 = sample("w_0", dist.Uniform(-100,100))
    w_1 = sample("w_1", dist.Uniform(-100,100))
    theta_dict = {"Omega_m":Omega_m, "w_0":w_0, "w_1":w_1}
    mu = cosm.dist_mod(theta_dict)
    
    sample("y", dist.MultivariateNormal(mu, cov), obs=y)


def model_alpha(x, y, cov):
    cosm = cosmo(0.7, x)
    Omega_m = sample("Omega_m", dist.Uniform(0,1))
    w_0 = sample("w_0", dist.Normal(-1.,5))
    w_1 = sample("w_1", dist.Normal(0.,5.))
    a = sample("alpha", dist.TruncatedNormal(0., 5.))
    theta_dict = {"Omega_m":Omega_m, "w_0":w_0, "w_1":w_1, "a":a}
    mu = cosm.dist_mod(theta_dict)
    
    sample("y", dist.MultivariateNormal(mu, cov), obs=y)
    
    
def model_alpha_planck(x, y, cov):
    cosm = cosmo(0.7, x)
    Omega_m = sample("Omega_m", dist.TruncatedNormal(0.3166,0.0084, low=0, high=1))
    w_0 = sample("w_0", dist.TruncatedNormal(-0.957, 0.08))
    w_1 = sample("w_1", dist.TruncatedNormal(-0.32, 0.29))
    a = sample("alpha", dist.TruncatedNormal(0., 0.5))
    theta_dict = {"Omega_m":Omega_m, "w_0":w_0, "w_1":w_1, "a":a}
    mu = cosm.dist_mod(theta_dict)
    
    sample("y", dist.MultivariateNormal(mu, cov), obs=y)
    
def model_alpha_noprior(x, y, cov):
    cosm = cosmo(0.7, x)
    Omega_m = sample("Omega_m", dist.Uniform(0,1))
    w_0 = sample("w_0", dist.Uniform(-100, 100))
    w_1 = sample("w_1", dist.Uniform(-100, 100))
    a = sample("alpha", dist.Uniform(-10, 10))
    theta_dict = {"Omega_m":Omega_m, "w_0":w_0, "w_1":w_1, "a":a}
    mu = cosm.dist_mod(theta_dict)
    
    sample("y", dist.MultivariateNormal(mu, cov), obs=y)


    
#define initial conditions
init_dict = {'init_0':{'Omega_m':0.5, 'w_0':0., 'w_1':1., 'a':1.}, 'init_1':{'Omega_m':0.3, 'w_0':-1., 'w_1':0., 'a':0.}}




models = [model_lcdm, model_lcdm_planck, 
          model_cpl, model_cpl_planck, model_cpl_noprior]#model_alpha, model_alpha_planck, model_alpha_noprior]

model_names = ['lcdm', 'lcdm',
               'cpl', 'cpl', 'cpl']
               #'alpha', 'alpha', 'alpha']

prior_names = ['prior0', 'prior1',
               'prior0', 'prior1', 'prior2']
               #'prior0', 'prior1', 'prior2']

dict_keys = ['model', 'name', 'prior']
results_dir='./numpyro_results/'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

model_dict = dict(zip(dict_keys, [models, model_names, prior_names]))

for i in range(len(models)):
    for j in ['init_0', 'init_1']:
        try:
            run_svi_manual(model_dict,i, x, y, cov, init_dict, j, 
                           num_samples, stepsize, grad_steps, intermediate_steps, results_dir=results_dir)
        except Exception as e:
            print('Exception raised!', e)



        
        

