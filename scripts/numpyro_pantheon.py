# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 17:15:20 2022

@author: jakob
"""


import emcee as em
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.lines as mlines
from cosmo_jnp import cosmo, pantheon

import jax
import jax.numpy as jnp
import jax.random as random

import numpyro
import numpyro.distributions as dist
from numpyro import sample, param, plate
import numpyro.optim as optim
from numpyro.diagnostics import hpdi, print_summary
from numpyro.infer import Predictive, SVI, Trace_ELBO, init_to_value
from numpyro.infer.autoguide import AutoMultivariateNormal, AutoLaplaceApproximation
from numpyro.infer.initialization import init_to_median, init_to_value

from optax import adam

from tqdm import tqdm

import pickle


p = pantheon('pantheon.txt', 'pantheon_covsys.txt')
p_inv = p.inv_cov
p_cov = jnp.linalg.inv(p_inv)

panth_dict = {'x' : p.redshift1, 'y' : p.dm, 'cov':p_cov, 'inv_cov' : p_inv, 'name':'panth'}

#defining cosmology object
cosmology = cosmo(0.7, p.redshift1)


def model_lcdm(x, y, cov):
    cosm = cosmo(0.7, x)
    Omega_m = sample("Omega_m", dist.Uniform(0,1))
    mu = cosm.dist_mod(Omega_m)
    
    sample("y", dist.MultivariateNormal(mu, cov), obs=y)
    
    
def model_lcdm_planck(x, y, cov):
    cosm = cosmo(0.7, x)
    Omega_m = sample("Omega_m", dist.TruncatedNormal(0.3166,0.0084, low=0, high=1))
    mu = cosm.dist_mod(Omega_m)
    
    sample("y", dist.MultivariateNormal(mu, cov), obs=y)

def model_cpl(x, y, cov):
    cosm = cosmo(0.7, x)
    Omega_m = sample("Omega_m", dist.Uniform(0,1))
    w_0 = sample("w_0", dist.TruncatedNormal(-1,5))
    w_1 = sample("w_1", dist.TruncatedNormal(0,5))
    mu = cosm.dist_mod(Omega_m, w_0, w_1)
    
    sample("y", dist.MultivariateNormal(mu, cov), obs=y)
    
def model_cpl_planck(x, y, cov):
    cosm = cosmo(0.7, x)
    Omega_m = sample("Omega_m", dist.TruncatedNormal(0.3166,0.0084, low=0, high=1))
    w_0 = sample("w_0", dist.TruncatedNormal(-0.957, 0.08))
    w_1 = sample("w_1", dist.TruncatedNormal(-0.32, 0.29))

    mu = cosm.dist_mod(Omega_m, w_0, w_1)
    
    sample("y", dist.MultivariateNormal(mu, cov), obs=y)
    
def model_cpl_noprior(x, y, cov):
    cosm = cosmo(0.7, x)
    Omega_m = sample("Omega_m", dist.Uniform(0,1))
    w_0 = sample("w_0", dist.Uniform(-100,100))
    w_1 = sample("w_1", dist.Uniform(-100,100))
    mu = cosm.dist_mod(Omega_m, w_0, w_1)
    
    sample("y", dist.MultivariateNormal(mu, cov), obs=y)


def model_alpha(x, y, cov):
    cosm = cosmo(0.7, x)
    Omega_m = sample("Omega_m", dist.Uniform(0,1))
    w_0 = sample("w_0", dist.Normal(-1.,5))
    w_1 = sample("w_1", dist.Normal(0.,5.))
    a = sample("alpha", dist.TruncatedNormal(0., 5.))
    
    mu = cosm.dist_mod(Omega_m, w_0, w_1, a)
    
    sample("y", dist.MultivariateNormal(mu, cov), obs=y)
    
    
def model_alpha_planck(x, y, cov):
    cosm = cosmo(0.7, x)
    Omega_m = sample("Omega_m", dist.TruncatedNormal(0.3166,0.0084, low=0, high=1))
    w_0 = sample("w_0", dist.TruncatedNormal(-0.957, 0.08, low=-2, high=0))
    w_1 = sample("w_1", dist.TruncatedNormal(-0.32, 0.29, low=-1, high=1))
    a = sample("alpha", dist.TruncatedNormal(0, 0.5, low=-1, high=1))
    
    mu = cosm.dist_mod(Omega_m, w_0, w_1, a)
    
    sample("y", dist.MultivariateNormal(mu, cov), obs=y)
    
def model_alpha_noprior(x, y, cov):
    cosm = cosmo(0.7, x)
    Omega_m = sample("Omega_m", dist.Uniform(0,1))
    w_0 = sample("w_0", dist.Uniform(-100, 100))
    w_1 = sample("w_1", dist.Uniform(-100, 100))
    a = sample("alpha", dist.Uniform(-10, 10))
    
    mu = cosm.dist_mod(Omega_m, w_0, w_1, a)
    
    sample("y", dist.MultivariateNormal(mu, cov), obs=y)


def run_svi_manual(model_dict, it, x, y, cov, initial_values, init, num_samples=10000, stepsize=0.001, grad_steps=5000):
    """
    Handles the complete inference process. Does not rely on the SVI.run()
    method but rather does it manually through jax.jit(SVI.update())
    which is more flexible but yields the same output at same speed.
    Takes intermediate results at logarithmic steps and saves samples,
    losses, params and quantiles in a .pickle file.

    Parameters
    ----------
    model_dict : dict, containing
        model:
            model which will be used, e.g. LCDM or CPL
            model does already contain the prior, so be specific in 
            the choice of the model
        name : str, 
            only needed for the name of the output file, e.g. 'lcdm'
        prior : str, 
            only needed for the name of the output file, 
            numerated, i.e. 'prior0', 'prior1' in a pre-defined order
    x : DeviceArray or ndarray, 
        redshift or x values
    y : DeviceArray or ndarray, 
        distance modulus or y values
        input to the 'obs' argument of the likelihood
    cov : DeviceArray or ndarray, 
        covariance of the observed distance
        modulus values
    initial_values : dict, 
        initial values for the optimizer passed in the
        Autoguide generator, if containing more entries then parameters
        the first in order are chosen
    init : str, 
        either 'init_0' or 'init_1' distinguishes between un- and biased
        initial values to study the influence to the outcome, 
        also passed to the filename
    num_samples : int, optional
        number of samples from the approximated posterior. The default is 10000.
    stepsize : float, optional
        Stepsize of the optimizer. A bigger values means that the optimizer
        converges faster, but is also likeli to produce missleading results.
        The default is 0.001.
    grad_steps : int, optional
        Number of gradient steps the optimizer does. The default is 5000.

    Returns
    -------
    dict
        Returns a dictionary of dictionarys and lists.
        sample_dict: dict, containing samples taken at each intermediate step
        losses: list, total ELBO losses.
        params: dict, guide params at each intermediate step
        quantiles: dict, mean and 16% and 84% values at each intermediate step
            of the true underlying variables

    """
    
    
    
    model = model_dict['model'][it]
    name_model = model_dict['name'][it]
    name_prior = model_dict['prior'][it]
    
    filename = name_model+'_'+name_prior+'_'+init+'_'+str(stepsize)+'.pickle'

    init_vals = initial_values[init]
    
    def func(state):
        state, loss = svi.update(state)
        return state, loss
    
    guide = AutoMultivariateNormal(model, init_loc_fn=init_to_value(values=init_vals))
        
    svi = SVI(model, 
        guide, 
        adam(stepsize), 
        Trace_ELBO(),     
        x = x,
        y = y,
        cov = cov
    )
    
    state, loss = svi.stable_update(svi.init(random.PRNGKey(3141)))
    
    steps = [0]
    steps += [100*int(i/100) for i in jnp.geomspace(2,grad_steps,7)]
    steps_diff = jnp.diff(jnp.array(steps))
    
    index = 0
    param_dict = {}
    quant_dict = {}
    
    losses = []
    
    print(filename+' startet...')
    
    for i in tqdm(range(grad_steps)):
        if i in steps_diff:
            par = svi.get_params(state)
            param_dict['param_{}'.format(index)] = par
            quant = guide.quantiles(par, [0.16, 0.5, 0.84])
            quant_dict['mean_{}'.format(index)] = quant
            index += 1
        else: 
            pass
        state, loss = jax.jit(func)(state)
        losses += [loss]
    
    samples_dict = {}
    for k in param_dict.keys():
        samples = guide.sample_posterior(random.PRNGKey(1235), param_dict[k], (num_samples,))
        samples_dict[k] = samples
        
    
    res_dict = {'samples':samples_dict, 'losses':losses, 'params':param_dict, 'quantiles':quant_dict}
    
    with open(filename, 'wb') as handle:
        pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    print('numpyro_results/'+filename+' saved!')

    return res_dict
    

init_dict = {'init_0':{'Omega_m':0.5, 'w_0':0., 'w_1':1., 'a':1.}, 'init_1':{'Omega_m':0.3, 'w_0':-1., 'w_1':0., 'a':0.}}


x = p.redshift1
y = p.dm
cov = p_cov

num_samples = 100
grad_steps = 100
stepsize = 0.01

models = [model_lcdm, model_lcdm_planck, 
          model_cpl, model_cpl_planck, model_cpl_noprior,
          model_alpha, model_alpha_planck, model_alpha_noprior]

model_names = ['lcdm', 'lcdm',
               'cpl', 'cpl', 'cpl',
               'alpha', 'alpha', 'alpha']

prior_names = ['prior0', 'prior1',
               'prior0', 'prior1', 'prior2',
               'prior0', 'prior1', 'prior2']

dict_keys = ['model', 'name', 'prior']


model_dict = dict(zip(dict_keys, [models, model_names, prior_names]))

for i in range(len(models)):
    for j in ['init0', 'init1']:
        try:
            run_svi_manual(model_dict,i, x, y, cov, init_dict, j, 
                           num_samples, stepsize, grad_steps)
        except Exception:
            print('Exception raised!')



        
        

