# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 00:00:46 2022

@author: jakob
"""

import arviz as az
import os
import matplotlib.pyplot as plt
import pickle

for root, dirs, files in os.walk('emcee_results/'):
    for file in files:
        idata = az.from_netcdf('emcee_results/'+str(file))
        az.plot_trace(idata)
        plt.tight_layout()
        
for root, dirs, files in os.walk('numpyro_results/'):
    for file in files:
        with open('numpyro_results/'+str(file), 'rb') as f:
            res_dict = pickle.load(f)
            plt.figure()
            plt.plot(res_dict['losses'])