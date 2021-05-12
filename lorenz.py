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
    def __init__(self,dt,param,X0,n_iter,t0=0.,scheme='euler') :
        self.n_iter = n_iter
        self.dt = dt # time
        self.t0 = t0 # initial time
        self.scheme = scheme
        # parameters of the model sigma, rho, beta in order
        self.parameters = param
        # initial condition
        self.x0 = X0
        # simulation parameters
        
        # results storage
        self.time = self.t0 # time of simulation

        self.time_series = [self.time] # time series
        self.xvar = np.copy(self.x0) # current position vector
        self.xvar_series = np.zeros((self.n_iter,3)) # storage of every position for each time step
        self.xvar_series[0][:] = self.x0 
    
    def __repr__(self) :
        return f'coordinates , x: {self.xvar[0]}, y= {self.xvar[1]}, z: {self.xvar[2]}'
    
    def time_to_it(self,t) :
        '''
        convert time in iteration
        '''
        return int(round(t,3)/self.dt)
    
    def it_to_time(self,i) :
        '''
        convert iteration in time
        '''
        return self.dt * i

    def rhs(self,x=None) :
        '''
        return the temporal derivative of the position vector, if no vector given, takes self.xvar
         - xout : array(dx/dt,dy/dt,dz/dt)
        '''
        # allocation of xout
        xout = np.zeros(3)
        if not np.all(x) :
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
        if self.scheme == 'euler' :
            return self.euler_step(x)
        
        elif self.scheme == 'RK4' :
            return self.RK4_step(x)
        
        else :
            print('not implemented yet')
    
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
        dxout[0] = dx[0] + self.dt*self.parameters[0]*(dx[1]-dx[0])
        dxout[1] = dx[1] + self.dt*((self.parameters[1]-x[2])*dx[0] - (dx[1]+x[0]*dx[2]))
        dxout[2] = dx[2] + self.dt*(x[1]*dx[0]+x[0]*dx[1]-self.parameters[2]*dx[2])
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
        u_out[0] = u_adj[0] + self.dt*((self.parameters[1]-x[2])*u_adj[1]+x[1]*u_adj[2]-self.parameters[0]*u_adj[0])
        u_out[1] = u_adj[1] + self.dt*(self.parameters[0]*u_adj[0]-u_adj[1]+x[0]*u_adj[2])
        u_out[2] = u_adj[2] - self.dt*(x[0]*u_adj[1]+self.parameters[2]*u_adj[2])
        return u_out


    
    def forward(self,niter,start=1) :
        '''
        run a niter iteration simulation of the lorenz system using a forward scheme
        '''
        for i in range(start,start+niter) :
            self.xvar = self.step(self.xvar)
            self.xvar_series[i] = self.xvar
            self.time_series.append(self.time)
    
    def re_initiate(self) :
        '''
        reinitiate the model, time = 0 and empty the time and positions series
        '''
        self.time = self.t0
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
    
    def euler_step(self,x) :
        xout = np.zeros(3) # allocate the output vector
        xout[:] = x + self.dt*self.rhs(x)
        self.time += self.dt # update time
        return xout
    
    def RK4_step(self,x) :
        '''
        use a 4th order Runge-Kutta scheme 
        '''
        k1 = self.rhs(x)
        k2 = self.rhs(x+0.5*self.dt*k1)
        k3 = self.rhs(x+0.5*self.dt*k2)
        k4 = self.rhs(x+self.dt*k3)
        xout = x + (self.dt/6)*(k1+2*(k2+k3)+k4)
        self.time += self.dt # update time
        return xout
    
    def RK4_step_tan(self,x,dx) :
        '''
        tangent step with a RK4 scheme
        '''
        dxout = np.zeros(3)
    
    
    
