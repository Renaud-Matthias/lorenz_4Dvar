#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 31 15:40:15 2021

@author: renamatt
"""

from lorenz import *
from obs import *
from ana import *
from scipy.optimize import minimize
import numpy as np


def assimil(n_window,n_step,n_assimil,n_simul,dt,param_true,param_assimil,n_sub,X0,Xb,scheme=None) :
    '''
    Run multiple assimilation window from time 0 to final time
    PARMETERS :
        - n_window , size of assimilation window
        - n_step , number of iteration between two assimilation, windows cross each other
        - n_assimil , number of window
        - n_simul , size of the simulation
        - n_sub , number of iterations between two observations
        - Xb , background state for first window
    RETURNS :
        - M_true : true model with the true trajectory
        - M_ana : the model contening the analysed trajectory
    '''
    # create and run true model
    M_true = Model(dt,param_true,X0,n_simul,scheme=scheme)
    M_true.forward(n_simul-1)
    
    # create obs operator
    Obs = create_Obs(M_true, n_simul, n_sub)
    R = Obs.std*np.eye(3) # observation error covariance matrix
    B = np.eye(3) # background error covariance matrix
    
    # create analysed model
    M_ana = Model(dt,param_assimil,Xb,n_simul)
    for k in range(0,n_assimil) :
        print(f'\n** window {k+1} **\n')
        first_window = k==0 # indicate when first assimilation window is running
        if first_window :
            x_res = ana_4Dvar(dt,param_assimil,n_window,Xb=Xb,R=R,Obs=Obs)
        else :
            Xb = np.copy(M_ana.xvar_series[k*n_step])
            x_res = ana_4Dvar(dt,param_assimil,n_window,Xb=Xb,R=R,B=B,Obs=Obs,i0=k*n_step)
        
        M_ana.xvar = x_res
        
        last_window = k==n_assimil-1 # indicate when last assimilation window reached
        
        start = k*n_step
        if last_window :
            M_ana.forward(n_window,start=start)
        else :
            M_ana.forward(n_step,start=start)
    
    return M_true,M_ana, Obs


def ana_4Dvar(dt,param,n_iter,Xb=None,B=None,R=None,Obs=None,i0=0) :
    '''
    Perform a 4Dvar assimilation and return the result of the assimilation
    PARAMETERS :
        - dt , time step
        - param , parameters of the lorenz model
        - n_iter , number of iteration in the assimilation
        - Xb , background state
        - B , background covariance error matrix
        - R , observation covariance error matrix
        - Obs , Observation object, contain the observations and the observation operators
    RETURN :
        - Xout , result of the assimilation, initial coordinates that best fits observations and background
    '''
    # model to analyse
    M = Model(dt,param,np.ones(3),n_iter)
    Var = Variational(Xb=Xb,B=B,R=R,M=M,Obs=Obs,i0=i0)
    X0 = np.copy(Xb)
    res = minimize(Var.cost,X0,options={'disp':True},jac=Var.grad)
    return res.x


def create_Obs(M_true,n_simul,n_sub) :
    '''
    Creates Observation object and generates the observations within the object
    '''
    Obs = Observation(n_simul,n_sub) # create Observation object
    Obs.gen_obs(M_true) # generate observation
    return Obs

    
