# -*- coding: utf-8 -*-
"""
Created on Fri May 20 20:34:57 2022

@author: jakob
"""

import jax.numpy as jnp
import numpy as np
from scipy.constants import c
import jax_cosmo as jc

class cosmo:
    '''defines a cosmology. The redshifts are handed over
    so that the array needed for the integral computation 
    is precomputed. The value of H_0 is in units of km/Mpc/s'''
    
    def __init__(self, h, z):
        '''h: value of h. type float.
        z: array of redshifts.'''
        
        self.H_0 = 100 * h * 1000 / c
        self.z_arr = jnp.linspace(0., z, 500)
        self.z = z

    	
    def H_z(self, Omega_m, w_0=-1, w_1=0, a=0):
        '''Omega_m, w_0, w_1, a: cosmological parameters, type float
        returns a matrix of length len(z) times z_sample.'''
        
        H_z = (self.H_0 * jnp.reciprocal(jnp.exp(jnp.log(1+self.z_arr)*a))) * jnp.sqrt(Omega_m * (1+self.z_arr) ** 3 + 
                                                       (1 - Omega_m)*(1+self.z_arr)**(3*(1+w_0+w_1))*jnp.exp(-3*w_1/(1+self.z_arr)))
        return H_z
    
    def d_L(self, H):
        '''The luminosity distance is computed via an integral. H is a matrix.
        It is integrated over axis 0 so that the remainder is an array
        of lenght len(z) and can be element-wise multiplied with (1+z).
        returns array of length len(z)'''
        
        d_L = (1+self.z) * jnp.trapz(1/H, x=self.z_arr, axis=0)
        return d_L
        
    def dist_mod(self, theta_dict):
        '''returns the theoretical predicted distance modulus'''
        H_of_z  = self.H_z(**theta_dict)
        dist_L = self.d_L(H_of_z)
        dist_mod = 25 + 5 * jnp.log10(dist_L)
        return dist_mod

    
class cosmo_jc:
    def __init__(self, h, z):
        self.a = jc.utils.z2a(z)
        self.h = h
        
    def dist_mod(self, theta_dict):
        '''returns the theoretical predicted distance modulus'''
        
        Omega_b = 0.05
        Omega_c = theta_dict.get('Omega_m', 0.3) - Omega_b
        w0 = theta_dict.get('w_0', -1)
        w1 = theta_dict.get('w_1', 0)
        cosmology = jc.Cosmology(h=self.h, Omega_c = Omega_c, Omega_b=Omega_b, w0=w0, wa=w1, Omega_k=0, n_s=0.96, sigma8=0.83)
        dist_L = (jc.background.angular_diameter_distance(cosmology, self.a)/self.a**2)/self.h
        dist_mod = 25 + 5 * jnp.log10(dist_L)
        return dist_mod
    

class union:
    '''data class for the union 2.1 data. The columns 2-5 are used
    which contain redshift, distance modulus and distance modulus error
    as well as the probability of the host galaxy to be of high mass.
    But this property is not important. The covariance is imported 
    and directly inverted.'''
    def __init__(self, name, cov):
        matrix1 = np.genfromtxt(name, delimiter='\t', skip_header=5, usecols=(1,2,3,4))
        matrix = jnp.asarray(matrix1)
        self.redshift = matrix[:,0]
        self.dm = matrix[:,1]
        self.err = matrix[:,2]
        self.prob = matrix[:,3]
        
        covariance = np.genfromtxt(cov, delimiter='\t', usecols=(range(len(self.redshift))))
        self.cov = covariance
        self.inv_cov = jnp.asarray(np.linalg.inv(covariance))
        
        
class pantheon:
    '''data class for pantheon data. Only four columns are used.
    The covarince matrix is passed as a list of size 1048x1048
    and can be reshaped as a matrix.'''
    def __init__(self, name, cov):
        matrix1 = np.genfromtxt(name, delimiter='', skip_header=7)
        matrix = jnp.asarray(matrix1)
        self.redshift1 = matrix[:,7]
        self.redshift2 = matrix[:,9]
        self.dm = matrix[:,40]
        self.err =matrix[:,42]
        covariance = np.loadtxt(cov)
        cov_shape = int(covariance[0])
        cov = covariance[1:].reshape((cov_shape, cov_shape)) + np.diag(self.err**2)
        self.cov = cov 
        self.inv_cov = jnp.asarray(np.linalg.inv(cov))

        

        



