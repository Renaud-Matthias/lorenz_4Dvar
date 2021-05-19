#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 10:20:39 2021

@author: Matthias
"""

import numpy as np
import matplotlib.pyplot as plt

class Model :
    '''
    simulate a Lorenz system for one set of initial condition
     INPUTS :
         - dt : temporal discretisation, 0.01 s default
         - param : size 3 array contening the parameters sigma rho and beta
    '''
    def __init__(self,dt,param,X0,n_iter,t0=0.,scheme='euler',test=False) :
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
        
        if test :
            self.test_tan()
    
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
        if self.scheme == 'euler' :
            return self.step_tan_euler(x, dx)
        elif self.scheme == 'RK4' :
            return self.step_tan_RK4(x, dx)
        else :
            print(f'{self.scheme} model not implemented')
    
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


#####################
# MODEL PROPAGATION #
#####################

# Euler scheme

    def euler_step(self,x) :
        xout = np.zeros(3) # allocate the output vector
        xout[:] = x + self.dt*self.rhs(x)
        self.time += self.dt # update time
        return xout
    
    def step_tan_euler(self,x,dx) :
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

# RK4 scheme

    def coef_RK4(self,x) :
        '''
        return the coef k1,k2,k3,k4 from the RK4 scheme
        '''
        k1 = self.rhs(x)
        k2 = self.rhs(x+0.5*self.dt*k1)
        k3 = self.rhs(x+0.5*self.dt*k2)
        k4 = self.rhs(x+self.dt*k3)
        return k1,k2,k3,k4
    
    def RK4_step(self,x) :
        '''
        use a 4th order Runge-Kutta scheme 
        '''
        k1,k2,k3,k4 = self.coef_RK4(x)
        xout = x + (self.dt/6)*(k1+2*(k2+k3)+k4)
        self.time += self.dt # update time
        return xout
    
    def step_tan_RK4(self,x,dx) :
        '''
        tangent step with a RK4 scheme
        '''
        K1,K2,K3,K4 = self.jac_coef_RK4(x,dx)
        dxout = dx + (self.dt/6)*(K1+ 2*K2 + 2*K3 + K4)
        return dxout
    
    
    def jac_rhs(self,X,dX) :
        '''
        Return the jacobian*dX of the right hand side term of the lorenz equation
        '''
        xout = np.zeros(3)
        sig,rho,bet = self.parameters[0],self.parameters[1],self.parameters[2]
        xout[0] = sig*(dX[1] - dX[0])
        xout[1] = (rho-X[2])*dX[0] - dX[1] - X[0]*dX[2]
        xout[2] = X[1]*dX[0] - X[0]*dX[1] - bet*dX[2]
        return xout

    def jac_coef_RK4(self,X,dX) :
        '''
        Return the Jacobian.dX of the RK4 coefficient k1,k2,k3,k4 at coordinate X
        '''
        k1,k2,k3,k4 = self.coef_RK4(X)
        # jacobian*dX of k1
        K1 = self.jac_rhs(X,dX)
        # jacobian*dX of k2
        dK2 = dX + (self.dt/2)*K1
        K2 = np.dot(self.jac_rhs(X+(self.dt/2)*k1),dK2)
        # jacobian*dX of k3
        dK3 = dX + (self.dt/2)*K2
        K3 = np.dot(self.jac_rhs(X+(self.dt/2)*k2),dK3)
        # jacobian*dX of k4
        dK4 = dX + self.dt*K3
        K4 = np.dot(self.jac_rhs(X+self.dt*k3),dK4)
        return K1,K2,K3,K4


#########################
# TEST OF TANGENT MODEL #
#########################


    def test_tan(self) :
        '''
        test if the tangent model is accurate
        '''
        print('\n** TANGENT TEST **\npass if result tend to 0\n\nvalueof coef :\t result\n')
        X = 10*np.random.random(3)
        dX = np.random.random(3)
        MX = self.step(X) # value of M(X) the model with coordinates X
        coef = 1.
        for i in range(10) :
            MXdX, tanDX = self.step(X+coef*dX), self.step_tan(X,coef*dX)
            res = abs(1 - np.linalg.norm(MXdX - MX)/np.linalg.norm(tanDX))
            print(f'{coef:2E} : {res:2E}')
            coef = 0.1*coef
        





