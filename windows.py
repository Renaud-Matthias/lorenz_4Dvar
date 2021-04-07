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


def assimil(n_window,n_step,n_assimil,n_simul,dt,param_true,param_assimil,n_sub,X0,Xb,Pb,R) :
    '''
    Run multiple assimilation window from time 0 to final time
    PARMETERS :
        - n_window , size of assimilation window
        - n_step , number of iteration between two assimilation, windows cross each other
        - n_assimil , number of window
        - n_simul , size of the simulation
        - n_sub , number of iterations between two observations
        - Xb , background state for first window
        - Pb , background covariance error matrix
        - R , observation covariance error matrix
    RETURNS :
        - M_true : true model with the true trajectory
        - M_ana : the model contening the analysed trajectory
    '''
    # create and run true model
    M_true = Model(dt,param_true,X0,n_simul)
    M_true.forward(n_simul)
    
    for k in range(0,n_assimil) :
        if k == 0 :
            M_obs = create_Model_obs(M_true, n_window, n_step, k)
            Obs = create_Obs(M_obs, n_window, n_sub,first=True)
            x_res = ana_4Dvar(dt,param_assimil,n_window,Xb,Pb,R,Obs) # result of first assimilation
            M_ana = Model(dt,param_assimil,x_res,n_simul) # result model (analysed)
            M_ana.forward(n_step)
        else :
            M_obs = create_Model_obs(M_true, n_window, n_step, k)
            Obs = create_Obs(M_obs, n_window, n_sub)
            # result of assimilation
            M_ana.xvar = ana_4Dvar(dt,param_assimil,n_window,M_ana.xvar,Pb,R,Obs) # result of assimilation
            M_ana.xvar_series[k*n_step] = M_ana.xvar
            run_window(M_ana,k,n_window,n_step,n_assimil)
    
    return M_true,M_ana


def ana_4Dvar(dt,param,n_iter,Xb,Pb,R,Obs) :
    '''
    Perform a 4Dvar assimilation and return the result of the assimilation
    PARAMETERS :
        - dt , time step
        - param , parameters of the lorenz model
        - n_iter , number of iteration in the assimilation
        - Xb , background state
        - Pb , background covariance error matrix
        - R , observation covariance error matrix
        - Obs , Observation object, contain the observations and the observation operators
    RETURN :
        - Xout , result of the assimilation, initial coordinates that best fits observations and background
    '''
    # model to analyse
    M = Model(dt,param,np.ones(3),n_iter)
    Var = Variational(Xb,Pb,R,M,Obs)
    res = minimize(Var.cost,np.zeros(3),jac=Var.grad)
    return res.x


def create_Model_obs(M_true,n_window,n_step,k_wdw) :
    '''
    Creates Model used to get observations
    PARAMETERS :
        - M_true ,
        - n_window ,
        - n_step ,
        - k_wdw , nummer of the current assimilation window
    RETURNS :
        - M_obs , model corresponding to the window observations
    '''
    M_obs = Model(M_true.dt, M_true.parameters, M_true.xvar_series[k_wdw*n_step], n_window) # model to generate observation
    M_obs.forward(n_window) # run model
    return M_obs

def create_Obs(M_obs,n_window,n_sub,first=False) :
    '''
    Creates Observation object and generates the observations within the object
    '''
    # time at which observations are available, 0 indicates the start of the window and not the whole assimilation
    if first :
        T_obs = [i*n_sub*M_obs.dt for i in range(1,int(n_window/n_sub))]
    else :
        T_obs = [i*n_sub*M_obs.dt for i in range(0,int(n_window/n_sub))]
    Obs = Observation(T_obs,n_window) # create Observation object
    Obs.gen_obs(M_obs) # generate observation
    return Obs
    

def run_window(M_ana,k_wdw,n_window,n_step,n_assimil) :
    '''
    Once the assimimilation is done, make the analysed model run toward the next window
    '''
    if k_wdw < n_assimil-1 :
        M_ana.forward(n_step+1,start=k_wdw*n_step+1) # solution on new assimilation window
    else :
        M_ana.forward(n_window,start=k_wdw*n_step+1)

    
    
    


