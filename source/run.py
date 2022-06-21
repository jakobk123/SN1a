# -*- coding: utf-8 -*-
"""
Created on Mon Jun 20 22:57:28 2022

@author: jakob
"""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer import init_to_value, Trace_ELBO, SVI
from optax import adam
from tqdm import tqdm
import pickle
import emcee as em
import arviz as az



#emcee run_sampler
def run_sampler(dictionary, key, nwalkers,  dm, data_inv, cosm, num, name, prob_dict, prob_key, results_dir=''):
    
    initial = dictionary[key]
    log_probability = prob_dict[prob_key]
    
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
    
    file = idata.to_netcdf(results_dir+filename+'.nc', groups='posterior')
    
    return idata

def run_svi_manual(model_dict, it, x, y, cov, initial_values, init, 
                   num_samples=10000, stepsize=0.001, grad_steps=5000, 
                   intermediate_steps=2, results_dir='./', save=True):
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
    
    filename = results_dir+name_model+'_'+name_prior+'_'+init+'_'+str(stepsize)+'.pickle'

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
    steps += [100*int(i/100) for i in jnp.geomspace(1,grad_steps,intermediate_steps)]
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
    
    if save == True:
        with open(filename, 'wb') as handle:
            pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        print(filename+' saved!')
    else:
        pass

    return res_dict