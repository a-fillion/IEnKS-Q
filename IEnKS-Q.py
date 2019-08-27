#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fev 14 2019

@author: A. Fillion

A python implementation of the IEnKS-Q of "A. Fillion, M. Bocquet, S. Gratton, P. Sakov and S. Gürol, An iterative ensemble Kalman smoother in presence of additive model error, SIAM-JUQ, (2019). Manucript under review.

Cofunded by CEREA, joint laboratory École des Ponts ParisTech and EDF R&D and CERFACS.
"""

from numpy import zeros,ones,eye,copy,empty,newaxis,outer,mean,diag
from numpy.random import randn
from scipy.linalg import cholesky,solve_triangular,qr,norm
from scipy.sparse.linalg import eigsh,svds
from sys import stdout as std
from time import time


# Parameters ##################################################################
n = 40 # State space dimension
Nobs = 1000 # Total number of observation vector
L = 1 # DAW length
K = 1 # DAW first new observation vector
G = 0 # DAW last state prior
mb = 19 # first diagonal block size
mq = 40 # Model error ensemble size
Ndt = 1 # Number of model time step integration
b = 1 # First cycle state background std
q = (0.01*Ndt)**0.5 # model error std
r = 1 # Observation error std
jmax = 10 # Maximum number of Gauss-Newton iterations
crit = 10**-3 # Gaus-Newton convergence criterion
infl = 1.02 # Inflation
lin = "bundle" # Linearization type (bundle / transform)
prop = "lin" # propagation type (lin / nlin)
mod = "sp" # Model reduced evaluation type (full / sp / svd-I / svd-0)
msvd = 19 # Ensemble size in case of svd anomaly decomposition
if lin=="transform":
    eps=1
elif lin=="bundle":
    eps=10**-4 # Finite differences step
spin = 10**3 # Spin-up
H = lambda x : x # Observation operator
from lorenz96 import M as model # Model
###############################################################################

# Definitions #################################################################
S = L-K+1 # DAW shift
D = max(G,K-1) # first diagonal block index
m, m_ = mb+(L-D)*mq, mb+(L-D)*mq+S*mq # ensemble size, extended ensemble size
M, M_ = m+1, m_+1 # associated ensembles number of members
xt0 = 3*ones(n)+randn(n) # true state at t0
xt0 = model(xt0,spin*Ndt) # Spin-up
E,y = empty(((L+1)*n,M)),empty(S*n) # An ensemble and an observation vector
Vq = q*eye(n,mq) # Model error anomaly
def ortho(M,RR=False): # generates U random or not such as [1,U] orthogonal MxM
    if RR:
        U,_ = qr(randn(M,M)) # Random orthogonal matrix
    else:
        U = eye(M,M) # Not random orthogonal matrix
    v = U[:,0]-M**-0.5*ones(M)
    return (eye(M)-2*norm(v)**(-2)*outer(v,v))@U[:,1:] # Set U[:,0] to 1 with an Householder matrix
U = dict() # Dictionary of U to avoid recomputations
for s in range(0,2*S+1):
    ms = mb+s*mq
    U[ms] = ortho(ms+1)
U[msvd] = ortho(msvd+1)
def mdl(E,l): # Reduced model evaluation
    global nmod # Number of model evaluations
    m,M = E.shape[1]-1,E.shape[1]
    if mod=="full": # no reduction
        E0, nmod = model(E,Ndt), nmod+M
    if mod=="sp": # Reduction on the m0 first anomaly columns
        m0 = m_-(L+S-max(D,l))*mq
        M0 = m0+1
        V0,un,un0 = eye(m0,m), ones(M), ones(M0)
        E0 = E@(M**-1*outer(un,un0) + (m0/m)**0.5*U[m]@V0.T@U[m0].T)
        E0, nmod = model(E0,Ndt), nmod+M0
        E0 = E0@(M0**-1*outer(un0,un) + (m/m0)**0.5*U[m0]@V0@U[m].T)
    if mod=="svd-I": # svd based reduction, identity surrogate
        m0 = msvd
        M0 = m0+1
        un,un0 = ones(M), ones(M0)
        X = m**-0.5*E@U[m]
        U0,s,V0 = svds(X,m0)
        X0 = U0 @ diag(s)
        X1V1 = X - X0@V0
        E0 = mean(E,1)[:,newaxis]+m0**0.5*X0@U[m0].T
        E0, nmod = model(E0,Ndt), nmod+M0
        E0 = E0@(M0**-1*outer(un0,un)+(m/m0)**0.5*U[m0]@V0@U[m].T) +\
            m**0.5*X1V1@U[m].T
    if mod=="svd-0": # svd based reduction, null surrogate
        m0 = msvd
        M0 = m0+1
        un,un0 = ones(M), ones(M0)
        X = m**-0.5*E@U[m]
        U0,s,V0 = svds(X,m0)
        X0 = U0 @ diag(s)
        E0 = mean(E,1)[:,newaxis]+m0**0.5*X0@U[m0].T
        E0, nmod = model(E0,Ndt), nmod+M0
        E0 = E0@(M0**-1*outer(un0,un)+(m/m0)**0.5*U[m0]@V0@U[m].T)
    return E0
P = zeros((m,m)) # Permutation matrix
for l in range(m):
    P[l,m-1-l] = 1.
I = eye(m) # Identity matrix
S_RMSE, F_RMSE, nmod, = 0,0,0 # Displays
###############################################################################

# Assimilation ################################################################
t = time() # time counter
i = 0 # cycle counter
while i<Nobs:
    std.write("\r"+int(20*i/Nobs)*"="+">"+int(20*(Nobs-i-1)/Nobs)*"-"+str(int(100*(i+1)/Nobs))+"%") # Progress bar
    
    # Observations and first prior ensemble generation
    xt = copy(xt0)
    for l in range(0,L+1):
        if (i==0 and l<=G):
            E[l*n:(l+1)*n] = xt[:,newaxis] + b*randn(n,M)
        if (i==0 and l>G):
            E[l*n:(l+1)*n] = q*randn(n,M)
        if (i==0 and l==L):
            vb, Vb = mean(E,1), E@U[m]*m**(-0.5)
        if l>=K:
            y[(l-K)*n:(l-K+1)*n] = H(xt) + r*randn(n)
        if l==L:
            xtL = copy(xt)
        xt = model(xt,Ndt) + q*randn(n)
        if(l+1==S):
            xtS = copy(xt)

    # Analysis ################################################################
    w, W, iW = zeros(m), eps*I, eps**(-1)*I
    j, stop = 0, 1
    while (j<jmax and stop>crit): # Gauss-Newton loop
        g, C = copy(w), copy(I)
        E = (vb+Vb@w)[:,newaxis] + m**0.5*Vb@W@U[m].T
        for l in range(0,L+1):
            if l>G:
                E[l*n:(l+1)*n] += mdl(E[(l-1)*n:l*n],l-1)
            if l>=K:
                V = H(E[l*n:(l+1)*n])
                f, F = mean(V,1), m**(-0.5)*V@U[m]@iW # Operator linearization
                V = F.T*r**(-2)
                g += V@(f-y[(l-K)*n:(l-K+1)*n]) # Gradient incrementation
                C += V@F # Hessian incrementation
        V = cholesky(P@C@P,lower=True) # Cholesky factorization of the permuted Hessian
        if lin=="bundle": # The Gauss newton system is solved
            dw = solve_triangular(V,-P@g,lower=True) # using the triangular factors
            dw = solve_triangular(V.T,dw,lower=False)
            w += P@dw
        if lin=="transform":
            W,iW = P@solve_triangular(V,I,lower=True).T@P,P@V.T@P # Lower triangular factor of the Hessian and its inverse
            dw = -W@W.T@g # The GN system is solved with this factor
            w += dw
        stop = norm(dw)*m**(-0.5) # Normalized convergence criterion
        j = j+1
    ###########################################################################
    
    # Propagation #############################################################
    if lin=="bundle":
        V = cholesky(P@C@P,lower=True)
        W = P@solve_triangular(V,I,lower=True).T@P # Lower triangular factor of the inverse Hessian
    va, Va = zeros(((L+S+1)*n)), zeros(((L+S+1)*n,m_))
    va[:(L+1)*n], Va[:(L+1)*n,:m] = vb+Vb@w, Vb@W
    for l in range(1,S+1): # Statistics pre extension
        Va[(L+l)*n:(L+l+1)*n,m+(l-1)*mq:m+l*mq] = copy(Vq)
    Ea = va[:,newaxis] + m_**0.5*Va@U[m_].T
    for l in range(G+1,G+S+1): # G shift
        if prop=="nlin" or l>L: # state statistics update with the model
            Ea[l*n:(l+1)*n] += mdl(Ea[(l-1)*n:l*n],l-1)
        elif prop=="lin" and l<=L: # state statistics update with the analysis linearization
            Ea[l*n:(l+1)*n] = mean(E[l*n:(l+1)*n],1)[:,newaxis]+\
                (m_/m)**0.5*E[l*n:(l+1)*n]@U[m]@iW@W@eye(m,m_)@U[m_].T
    vb, V = mean(Ea[S*n:],1), Ea[S*n:]@U[m_]*m_**(-0.5) # Next cycle prior mean and deviation matrix
    s,V = eigsh(V[:(D+1)*n]@V[:(D+1)*n].T,mb,which="LM") # Size reduction of the first diagonal block
    Vb = zeros(((L+1)*n,m)) # Next cycle prior reduced deviation matrix
    Q,_ = qr(randn(mb,mb)) # Random orthogonal matrix
    Vb[:(D+1)*n,:mb] = infl*V@diag(s**0.5)@Q # inflation and random rotations
    for l in range(D+1,L+1): # statistics post extension
        Vb[l*n:(l+1)*n,mb+(l-D-1)*mq:mb+(l-D)*mq] = copy(Vq)
    i += S
    ###########################################################################

    S_RMSE += norm(xt0-mean(E[:n],1))*n**(-0.5) # Smoothing RMSE update
    F_RMSE += norm(xtL-mean(E[L*n:],1))*n**(-0.5) # Filtering RMSE update
    xt0 = copy(xtS) # Propagation of the truth
t = time()-t
###############################################################################


# Displays ####################################################################
std.write("\nSmoothing error = "+str(S_RMSE*S/Nobs))
std.write("\nFiltering error = "+str(F_RMSE*S/Nobs))
std.write("\nNumber of model evaluation = "+str(nmod/Nobs))
std.write("\ntime = "+str(t)+"\n")
###############################################################################
