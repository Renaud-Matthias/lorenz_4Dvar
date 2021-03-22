#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:53:09 2021

@author: Matthias
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import inv


class Variational :
    
    def __init__(self,Xb,Pb,R,M,Obs) :
        self.Xb = Xb # background state
        self.Pb = Pb # background covariance error matrix
        self.Rinv = inv(R) # observation covariance error matrix
        self.M = M # Lorenz model
        self.Obs = Obs # observations (class observation)
        
    
    def cost(self,X) :
        '''
        coast function to minimize
         INPUTS :
             - initial coordinates
         OUTPUTS :
             - Jout, value of the cost function, float
        '''
        # re initialize the model
        self.M.re_initiate()
        # background cost
        b_er = X-self.Xb # background error
        gb = np.dot(inv(self.Pb),b_er)
        Jb = 0.5*np.dot(b_er,gb)
        
        # observation cost
        Jo = 0.
        u = np.copy(X)
        u_trj = [] # list to store the trajectory
        for it in range(self.M.n_iter) :
            u_trj.append(u)
            if self.Obs.isobserved(self.M.time) :
                miss = self.Obs.misfit(self.M.time,u) # H.x -y
                Jo = Jo + np.dot(miss,np.dot(self.Rinv,miss))
            u = self.M.step(u) # forward step
        J = 0.5*Jo + Jb # total cost
        return J
    
    def grad(self,X) :
        '''
        compute the gradient of the cost function at coordinates X
        PARAMETERS :
         - X : size 3 array containing the coordinates (x,y,z) where the gradient has to be evaluated
        RETURN :
         - u_adj : the gradient of the cost function
        '''
        # re initialize the model
        self.M.re_initiate()
        # background cost
        b_er = X-self.Xb # background error
        
        u = np.copy(X)
        u_trj = [] # list to store the trajectory
        # forward run to store the particle trajectory
        for it in range(self.M.n_iter) :
            u_trj.append(u)
            if self.Obs.isobserved(self.M.time) :
                miss = self.Obs.misfit(self.M.time,u) # H.x -y
            u = self.M.step(u) # forward step
        
        t = self.Obs.time_obs[-1]
        u_adj = self.Obs.H[t] @ inv(self.R) @ self.Obs.misfit(t,u)
        # backward in time
        for t in self.Obs.time_obs[:-1][::-1] :
            u_adj = self.M.step_adj(X, u_adj)
            if self.Obs.isobserved(t) :
                inov = self.Obs.misfit(t)
                
                u_adj += self.Obs.H[t].T @ self.R.T @ inov
        # u_adj is the gradient of the cpst function at X
        u_adj += np.dot(inv(self.Pb),b_er)
        
        return u_adj





