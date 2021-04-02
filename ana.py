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
        for it in range(self.M.n_iter) :
            if self.Obs.isobserved(self.M.time) :
                miss = self.Obs.misfit(self.M.time,u) # H.x -y
                Jo = Jo + np.dot(miss,np.dot(self.Rinv,miss))
            u = self.M.step(u) # forward step
        J = 0.5*Jo + Jb # total cost
        return J
    
    def grad(self,X) :
        '''
        compute the gradient of the cost function at coordinates X using adjoint coding
        PARAMETERS :
         - X : size 3 array containing the coordinates (x,y,z) where the gradient has to be evaluated
        RETURN :
         - u_adj : the gradient of the cost function
        '''
        # re initialize the model
        self.M.re_initiate()
        # background cost
        b_er = X-self.Xb # background error
        t_last = self.Obs.time_obs[-1] # time of last observation
        dt = self.M.dt # time discretisation
        
        u = np.copy(X)
        u_trj = [u] # list to store the trajectory
        # forward run to store the particle trajectory, stop at last observation
        for t_it in (i for i in range(int(t_last/dt))) :
            u = self.M.step(u) # forward step, u at iteration t_it+1
            u_trj.append(u)
        # adjoint coding
        t_adj = t_last - dt # time of last observation
        u_adj = self.Obs.H[t_last].T @ self.Rinv @ self.Obs.misfit(t_last,u)
        # backward in time
        while round(t_adj,5) > 0. :
            u_adj = self.M.step_adj(u_trj[int(t_adj/dt)], u_adj) # adjoint step
            if self.Obs.isobserved(t_adj) :
                # observation term if it exist
                inov = self.Obs.misfit(t_adj,u_trj[int(t_adj/dt)]) # inovation
                u_adj += self.Obs.H[t_adj].T @ self.Rinv @ inov
            t_adj -= dt
            t_adj = round(t_adj,5)
        
        # add eventual obs at initial time
        t_ini = self.M.t0
        if self.Obs.isobserved(t_ini) :
            u_adj += self.Obs.H[t_ini].T @ self.Rinv @ inov
        # add background component
        u_adj += np.dot(inv(self.Pb),b_er)
        # u_adj is the gradient of the cpst function at X
        return u_adj


