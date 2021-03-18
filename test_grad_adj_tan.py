#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 14:46:57 2021

@author: Matthias
"""

from lorenz import *
from ana import *


dt = 0.01 # temporal discretisation
parameters = [10.,11.,5.] # true parameters of the model
n_simul = 30 # number of iteration of the simulation

condi_ini = np.array([1.,3.,2.]) # initial condition

# model
Lor = Model(dt,parameters,condi_ini,n_simul)


# test tan, need to tend to zero

X_state = 10*np.random.random(3)
dx = np.ones(3)

print('test tan model :\n')
coef = 1.
while coef > 1e-08 :
    print(np.mean(Lor.step(X_state+coef*dx)-Lor.step(X_state)-Lor.step_tan(X_state,coef*dx)))
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