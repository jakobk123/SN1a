# -*- coding: utf-8 -*-
"""
Created on Tue Jun 21 00:00:46 2022

@author: jakob
"""

import arviz as az
import os
import matplotlib.pyplot as plt
import pickle
from matplotlib import rc
import matplotlib.lines as mlines
import numpy as np


plt.style.use('default')
plt.rcParams['font.size'] = 12.0
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.linewidth'] = 0.4
plt.rcParams['lines.linewidth'] = 0.4
plt.rcParams['xtick.labelsize'] = 10
rc('text', usetex=True)



class emcee_result:
    

    def __init__(self, filename):
        idata = az.from_netcdf(filename)
        self.idata = idata
        self.dim = len(idata.posterior)
        
    def trace(self, axes=None):
        
        var_dict = {"Omega_m" : r"$\Omega_{m,0}$",
                   "w_0" :  r"$w_0$",
                   "w_1" : r"$w_1$",
                   "alpha" : r"$\alpha$"}   
        
        labeller = az.labels.MapLabeller(var_name_map=var_dict)
        

        az.plot_trace(self.idata, axes=axes, labeller=labeller)
        
        
class numpyro_result:
    
    def __init__(self, filename):
        with open(filename, 'rb') as f:
            res_dict = pickle.load(f) 
        self.losses = res_dict['losses']
        self.samples = res_dict['samples']
        self.params = res_dict['params']
        self.quantiles = res_dict['quantiles']
        
    def plot_losses(self, ax=None, kwargs=dict()):
        if ax:
            ax.plot(self.losses, **kwargs)
        else:
            plt.plot(self.losses, **kwargs)
            
    def plot_samples(self, ax=None, all=True):
        
        par_dict =  {'param_0': '0',
                     'param_1': '100',
                     'param_2': '200',
                     'param_3': '300',
                     'param_4': '500',
                     'param_5': '1000',
                     'param_6': '1700',
                     'param_7': '2900',
                     'param_8': '5000',
                     'param_9': '8600',
                     'param_10': '14600',
                     'param_11': '25000'}
    
        if all == True:
            color = iter(plt.cm.rainbow(np.linspace(0, 1, len(self.samples.values()))))
            handles = []
            for i in self.samples.keys():
                s = self.samples[i]
                c = next(color)
                O = s['Omega_m']
                az.plot_kde(O, plot_kwargs=dict(color=c), ax=ax)
                label=par_dict[i]
                h = mlines.Line2D([], [], color=c, label=label)
                handles += [h]
            plt.legend(handles=handles)
            
        else:
            s = self.samples['param_11']['Omega_m']
            az.plot_kde(s, ax=ax)
                
        

fig7, axes7 = plt.subplots(1,2, figsize=(7,3))

lcdm_ni0p0 = numpyro_result('numpyro_results/lcdm_prior0_init_0_0.001.pickle') 
lcdm_ni0p0.plot_losses(axes7[0], kwargs={'label':'initial 1'})  

lcdm_ni1p0 = numpyro_result('numpyro_results/lcdm_prior0_init_1_0.001.pickle') 
lcdm_ni1p0.plot_losses(axes7[0],kwargs={'label':'initial 2'})  

axes7[0].legend()
axes7[0].set_ylabel(r'$\mathrm{ELBO}$')
axes7[0].set_xlabel(r'$\mathrm{grad\; steps}$')
axes7[0].set_title(r'$\mathrm{Prior 1}$')

lcdm_ni0p1 = numpyro_result('numpyro_results/lcdm_prior1_init_0_0.001.pickle') 
lcdm_ni0p1.plot_losses(axes7[1], kwargs={'label':'initial 1'})  

lcdm_ni1p1 = numpyro_result('numpyro_results/lcdm_prior1_init_1_0.001.pickle') 
lcdm_ni1p1.plot_losses(axes7[1],kwargs={'label':'initial 2'})  

axes7[1].legend()
axes7[1].set_ylabel(r'$\mathrm{ELBO}$')
axes7[1].set_xlabel(r'$\mathrm{grad\; steps}$')
axes7[1].set_title(r'$\mathrm{Prior 2}$')
fig7.tight_layout()
fig7.savefig('../plots/lcdm_losses.pdf')


fig8, axes8 = plt.subplots(1,3, figsize=(9,3))
cpl_ni0p0 = numpyro_result('numpyro_results/cpl_prior0_init_0_0.001.pickle') 
cpl_ni0p0.plot_losses(axes8[0], kwargs={'label':'initial 1', 'linewidth':1.5})  

cpl_ni1p0 = numpyro_result('numpyro_results/cpl_prior0_init_1_0.001.pickle') 
cpl_ni1p0.plot_losses(axes8[0],kwargs={'label':'initial 2', 'alpha':0.6, 'linewidth':1.5})  

axes8[0].legend()
axes8[0].set_ylabel(r'$\mathrm{ELBO}$')
axes8[0].set_xlabel(r'$\mathrm{grad\; steps}$')
axes8[0].set_title(r'$\mathrm{Prior 1}$')

cpl_ni0p1 = numpyro_result('numpyro_results/cpl_prior1_init_0_0.001.pickle') 
cpl_ni0p1.plot_losses(axes8[1], kwargs={'label':'initial 1', 'linewidth':1.5})  

cpl_ni1p1 = numpyro_result('numpyro_results/cpl_prior1_init_1_0.001.pickle') 
cpl_ni1p1.plot_losses(axes8[1],kwargs={'label':'initial 2', 'alpha':0.6, 'linewidth':1.5})  

axes8[1].legend()
axes8[1].set_ylabel(r'$\mathrm{ELBO}$')
axes8[1].set_xlabel(r'$\mathrm{grad\; steps}$')
axes8[1].set_title(r'$\mathrm{Prior2}$')

cpl_ni0p2 = numpyro_result('numpyro_results/cpl_prior2_init_0_0.001.pickle') 
cpl_ni0p2.plot_losses(axes8[2], kwargs={'label':'initial 1', 'linewidth':1.5})  

cpl_ni1p2 = numpyro_result('numpyro_results/cpl_prior2_init_1_0.001.pickle') 
cpl_ni1p2.plot_losses(axes8[2],kwargs={'label':'initial 2', 'alpha':0.6, 'linewidth':1.5})  

axes8[2].legend()
axes8[2].set_ylabel(r'$\mathrm{ELBO}$')
axes8[2].set_xlabel(r'$\mathrm{grad\; steps}$')
axes8[2].set_title(r'$\mathrm{Prior3}$')



# =============================================================================
# 
# cpl_ei0p0 = emcee_result('emcee_results/cpl_init0_prior0.nc')
# cpl_ei0p1 = emcee_result('emcee_results/cpl_init0_prior1.nc')
# cpl_ei1p0 = emcee_result('emcee_results/cpl_init1_prior0.nc')
# cpl_ei1p1 = emcee_result('emcee_results/cpl_init1_prior1.nc')
# cpl_ei1p2 = emcee_result('emcee_results/cpl_init1_prior2.nc')
# 
# fig2, axes2 = plt.subplots(3,2, figsize=(5,5))
# cpl_ei0p0.trace(axes=axes2)
# plt.tight_layout()
# fig2.savefig('../plots/cpl_ei0p0_trace.pdf')
# 
# fig3, axes3 = plt.subplots(3,2, figsize=(5,5))
# cpl_ei0p1.trace(axes=axes3)
# plt.tight_layout()
# fig3.savefig('../plots/cpl_ei0p1_trace.pdf')
# 
# fig4, axes4 = plt.subplots(3,2, figsize=(5,5))
# cpl_ei1p0.trace(axes=axes4)
# plt.tight_layout()
# fig4.savefig('../plots/cpl_ei1p0_trace.pdf')
# 
# fig5, axes5 = plt.subplots(3,2, figsize=(5,5))
# cpl_ei1p1.trace(axes=axes5)
# plt.tight_layout()
# fig5.savefig('../plots/cpl_ei1p1_trace.pdf')
# 
# fig6, axes6 = plt.subplots(3,2, figsize=(5,5))
# cpl_ei1p2.trace(axes=axes6)
# plt.tight_layout()
# fig6.savefig('../plots/cpl_ei1p2_trace.pdf')
# 
# 
# 
# 
# 
# 
# var_dict = {"Omega_m" : r"$\Omega_{m,0}$",
#            "w_0" :  r"$w_0$",
#            "w_1" : r"$w_1$",
#            "alpha" : r"$\alpha$"}   
# labeller = az.labels.MapLabeller(var_name_map=var_dict)
# 
# lcdm_ei0p0 = az.from_netcdf('emcee_results/lcdm_init0_prior0.nc')
# lcdm_ei0p1 = az.from_netcdf('emcee_results/lcdm_init0_prior1.nc')
# lcdm_ei1p0 = az.from_netcdf('emcee_results/lcdm_init1_prior0.nc')
# lcdm_ei1p1 = az.from_netcdf('emcee_results/lcdm_init1_prior1.nc')
# 
# 
# 
# 
# fig1, axes1 = plt.subplots(4,2, sharey='col', sharex='col', figsize=(5.6, 8))
# 
# ax1 = fig1.add_subplot(411, frameon=False)
# ax1.set_title(r'$\Lambda\mathrm{CDM \; Prior 1 \; initial\; conditions 1}$')
# ax1.set_xticks([])
# ax1.set_yticks([])
# az.plot_trace(lcdm_ei0p0, axes=axes1[[0],:], labeller=labeller)
# 
# ax2 = fig1.add_subplot(412, frameon=False)
# ax2.set_title(r'$\Lambda\mathrm{CDM \; Prior 1 \; initial\; conditions 2}$')
# ax2.set_xticks([])
# ax2.set_yticks([])
# az.plot_trace(lcdm_ei1p0, axes=axes1[[1],:], labeller=labeller)
# 
# ax3 = fig1.add_subplot(413, frameon=False)
# ax3.set_title(r'$\Lambda\mathrm{CDM \; Prior 2 \; initial\; conditions 1}$')
# ax3.set_xticks([])
# ax3.set_yticks([])
# az.plot_trace(lcdm_ei0p1, axes=axes1[[2],:], labeller=labeller)
# 
# ax4 = fig1.add_subplot(414, frameon=False)
# ax4.set_title(r'$\Lambda\mathrm{CDM \; Prior 2 \; initial \; conditions 2}$')
# ax4.set_xticks([])
# ax4.set_yticks([])
# az.plot_trace(lcdm_ei1p1, axes=axes1[[3],:], labeller=labeller)
# 
# for a in axes1:
#     a[0].set_title('')
#     a[1].set_title('')
#     a[0].set_xlabel(r"$\Omega_{m,0}$")
#     a[1].set_ylabel(r"$\Omega_{m,0}$")
#         
# 
# fig1.tight_layout()
# fig1.savefig('../plots/lcdm_trace.pdf')
# 
# =============================================================================










