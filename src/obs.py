#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 14:14:18 2021

@author: Matthias
"""

import numpy as np
import random as rd


# need to define the time of observation

class Observation :
    
    def __init__(self,n_simul,n_sub,std=0.1) :
        '''
        Create a set of observation, each observation is on one coordinate only (random)
         INPUTS :
             - iter_obs : list of iteration where observation are available
        '''
        
        # list of all iteration where observations are available
        self.iter_obs = [i for i in range(0,n_simul,n_sub)]
        
        # standard deviation of the observation
        self.std = std
        # is made and the value of the observation
        self.obs = {}
        # dictionnary of array, key is iteration i and items the associated observation operator
        self.H = {}
        # dictionnary containing the observation operators
        
        self.n_simul = n_simul
        self.n_obs = len(self.iter_obs) # number of observation
        
        
    def gen_obs(self,model_ref) :
        '''
        generate the set of observation from the reference model, it need to be forwarded (model_ref.forward(n))
        '''
        for k in range(self.n_simul) :
            
            if self.isobserved(k) :
                
                self.obs[k] = np.copy(model_ref.xvar_series[k]) + np.random.normal(0.,self.std,3) # add the observation
                h = np.eye(3)
                self.H[k] = h   
        
            
        
    def isobserved(self,i) :
        '''
        return True if an observation is available at iteration i or false if not
        '''
        if i in self.iter_obs :
            return True
        else :
            return False
        
    def misfit(self,i,u) :
        '''
        compute the innovation H.x-y for a specific observation
        '''
        if self.isobserved(i) :
            return np.dot(self.H[i],u)-self.obs[i]
        else :
            print(f'no observation available at iteration {i}')








