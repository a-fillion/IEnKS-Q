# -*- coding: utf-8 -*-
"""
Created on Fev 14 2019

@author: A. Fillion

The lorenz95 model integrated with Runge-Kutta 4
"""

import numpy as np

# The différential equation is X'=f(X)
def f(X):
    F = 8.0
    nl = X.shape[0]
    b = np.arange(nl)
    Xp1 = X[(b+1)%nl]
    Xm1 = X[(b-1)%nl]
    Xm2 = X[(b-2)%nl]
    return -np.multiply(Xm2,Xm1)+np.multiply(Xm1,Xp1)-X+F

# Fourth order Runge-Kutta 4 scheme
def RK4(X,dt):
    k1 = f(X)
    k2 = f(X+(dt/2.0)*k1)
    k3 = f(X+(dt/2.0)*k2)
    k4 = f(X+dt*k3)
    return X+(dt/6.0)*(k1+2.0*k2+2.0*k3+k4)

# Resolvent of the differential equation over N time steps
def M(x0,N):
    x = x0
    dt = np.sign(N)*0.05 # time step
    for i in range(np.abs(N)):
        x = RK4(x,dt)
    return x