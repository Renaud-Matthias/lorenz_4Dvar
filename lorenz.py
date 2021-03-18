#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:20:39 2021

@author: Matthias
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Model :
    '''
    simulate a Lorenz system for one set of initial condition
     INPUTS :
         - dt : temporal discretisation, 0.01 s default
         - param : size 3 array contening the parameters sigma rho and beta
    '''
    def __init__(self,dt,param,X0,n_iter) :
        self.n_iter = n_iter
        self.dt = dt # time 
        # parameters of the model sigma, rho, beta in order
        self.parameters = param
        # initial condition
        self.x0 = X0
        # simulation parameters
        
        # results storage
        self.time = 0. # time of simulation

        self.time_series = [self.time] # time series
        self.xvar = np.copy(self.x0) # current position vector
        self.xvar_series = np.zeros((self.n_iter,3)) # storage of every position for each time step
        self.xvar_series[0][:] = self.x0 
    
    def __repr__(self) :
        return f'coordinates , x: {self.xvar[0]}, y= {self.xvar[1]}, z: {self.xvar[2]}'

    def rhs(self) :
        '''
        return the temporal derivative of the position vector
         - xout : array(dx/dt,dy/dt,dz/dt)
        '''
        # allocation of xout
        xout = np.zeros(3)
        x = self.xvar
        xout[0] = self.parameters[0]*(x[1]-x[0])
        xout[1] = x[0]*(self.parameters[1]-x[2])-x[1]
        xout[2] = x[0]*x[1]-self.parameters[2]*x[2]
        return xout
    
    def step(self,x) :
        '''
        one step forward
         - x : coordinate before the step
        the functions return the new coordinates after one time step
        '''
        xout = np.zeros(3) # allocate the output vectors
        xout[:] = x + self.dt*self.rhs()
        self.time += self.dt # update time
        return xout
    
    def step_tan(self,x,dx) :
        '''
        Step using the Jacobian of the model, a perturbation dx of the coordinates
        propagates
        PARAMETERS
         - x : value of parameters (x,y,z) at the point considered (size 3 array)
         - dx : value of the perturbation (dx,dy,dz) (size 3 array)
        RETURN
         - dxout : value of the perturbation (dx,dy,dz) at the next iteration
        '''
        dxout = np.zeros(3)
        dxout[0] = self.dt*self.parameters[0]*(dx[1]-dx[0]) + dx[0]
        dxout[1] = self.dt*((self.parameters[1]-x[2])*dx[0] + x[0]*dx[2])
        dxout[2] = self.dt*(x[1]*dx[0]+x[0]*dx[1]) + (1-self.parameters[2])*dx[2]
        return dxout
    
    def step_adj(self,x,u_adj) :
        '''
        Adjoint step of the tangent model
        PARAMETERS
         - x : value of the coordinates at the point considered
         - u_adj : value of the vector lambda
        RETURN
         - u_out : value of the vector lambda at next iteration
        '''
        u_out = np.zeros(3)
        par = [self.parameters[i] for i in range(3)]
        u_out[0] = (1-self.dt*par[0])*u_adj[0] + self.dt*(par[1]-x[2])*u_adj[1] + self.dt*x[1]*u_adj[2]
        u_out[1] = self.dt*(par[0]*u_adj[0] + x[0]*u_adj[2])
        u_out[2] = self.dt*x[0]*u_adj[1] + (1-par[2])*u_adj[2]
        return u_out


    
    def forward(self,niter) :
        '''
        run a niter iteration simulation of the lorenz system using a forward scheme
        '''
        for i in range(1,niter) :
            self.xvar = self.step(self.xvar)
            self.xvar_series[i] = self.xvar
            self.time_series.append(self.time)
    
    def re_initiate(self) :
        '''
        reinitiate the model, time = 0 and empty the time and positions series
        '''
        self.time = 0.
        self.time_series = [self.time]
        self.xvar[:] = self.x0
        self.xvar_series = np.zeros((self.n_iter,3))
        self.xvar_series[0][:] = self.x0 

    
    
    def plot_3D(self) :
        '''
        plot a 3D image of the series of position of the particule
        '''
        fig = plt.figure(figsize=(8,8)) # create figure
        ax = fig.gca(projection='3d')
        xvar_array = np.array(self.xvar_series)
        ax.plot(xvar_array[:,0], xvar_array[:,1], xvar_array[:,2])
        # legend
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Y Axis")
        ax.set_zlabel("Z Axis")
        ax.set_title("Lorenz Attractor")
        plt.show()
    
    def plot_separate(self,Label=None) :
        xvar_array = np.array(self.xvar_series)
        renamatt # create figure
        plt.subplot(3,1,1)
        plt.plot(self.time_series,xvar_array[:,0],label=Label)
        plt.subplot(3,1,2)
        plt.plot(self.time_series,xvar_array[:,1],label=Label)
        plt.subplot(3,1,3)
        plt.plot(self.time_series,xvar_array[:,2],label=Label)
        plt.show()
    
    
    
