#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:46:57 2021

@author: Matthias
"""

from lorenz import *
from ana import *
from obs import *


dt = 0.01 # temporal discretisation
parameters = [10.,11.,5.] # true parameters of the model
n_simul = 30 # number of iteration of the simulation

# background state
Xb = np.array([0.,0.,0.])
# background covariance error matrix
sigma_b = 1.
Pb = sigma_b*np.eye(3)
# observation covariance error matrix
sigma_y = 0.1
R = sigma_y*np.eye(3)

condi_ini = np.array([1.,3.,2.]) # initial condition

# model
Lor = Model(dt,parameters,condi_ini,n_simul)
Lor.forward(n_simul)

# observation

n_sub = 5 # number of iteration between two observation

T_obs = [i*n_sub*dt for i in range(1,n_simul//n_sub)] # observation time

Obs = Observation(T_obs,n_simul)

Obs.gen_obs(Lor)

# Variational

Var = Variational(Xb,Pb,R,Lor,Obs)

# test tan, need to tend to zero

X_state = 10*np.random.random(3)
dx = np.ones(3)

print('test tan model :\n')
coef = 1.
while coef > 1e-08 :
    print((Lor.step(X_state+coef*dx)-Lor.step(X_state))/(Lor.step_tan(X_state,coef*dx)))
    coef = coef/10
print('\n')

# test adjoint

X = np.random.random(3)
Y = np.random.random(3)

prod1 = np.dot(Lor.step_tan(X_state,X),Y)

prod2 = np.dot(Lor.step_adj(X_state,Y),X)

print('test adjoint ok :',round(prod1,8)==round(prod2,8))
print('\n')


# test grad, need to tend to one

print('test gradient :\n')

X = np.array([10.,7.,2.1])
dx = np.ones(3)

coef = 1.
while coef > 1e-8 :
    print((Var.cost(X+coef*dx)-Var.cost(X))/np.dot(Var.grad(X).T,coef*dx))
    coef = coef/10

