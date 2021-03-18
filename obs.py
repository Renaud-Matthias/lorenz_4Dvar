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
    
    def __init__(self,T,n_simul) :
        '''
        Create a set of observation, each observation is on one coordinate only (random)
         INPUTS :
             - T : list of time where observation are available
        '''
        # list of all the time where an observation is available
        self.time_obs = T
        # is made and the value of the observation
        self.obs = {}
        # dictionnary of array, key is time t and items the associated observation operator
        self.H = {}
        # dictionnary containing the observation operators
        
        self.n_simul = n_simul
        self.n_obs = len(T) # number of observation
        
        
    def gen_obs(self,model_ref) :
        '''
        generate the set of observation from the reference model
        '''
        for k in range(self.n_simul) :
            t = model_ref.time_series[k]
            
            if self.isobserved(t) :
                
                self.obs[t] = np.copy(model_ref.xvar_series[k]) # add the observation
                h = np.eye(3)
                self.H[t] = h            
        
            
        
        
    def isobserved(self,t) :
        '''
        return True if an observation is available at time t or false if not
        '''
        if round(t,2) in self.time_obs :
            return True
        else :
            return False
        
    def misfit(self,t,u) :
        '''
        compute the innovation H.x-y for a specific observation
        '''
        if self.isobserved(t) :
            return np.dot(self.H[t],u)-self.obs[t]








