#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# Date: 24-09-2021
# Author: Pierre Roux
# Collaboration with Alexis Leculier

""" Code associated with the article " "
    This code contains the parameters and functions used to produce the figures of the article. The test section contains some of the instructions used to produce the figures.
"""

__author__ = "Pierre Roux"
__copyright__ = "Copyright 2007, The Cogent Project"
__credits__ = ["Pierre Roux", "Alexis Leculier"]
__license__ = "GNU General public license v3.0"
__version__ = "2.0"
__maintainer__ = "Pierre Roux"
__email__ = "pierre.rouxmp@laposte.net"
__status__ = "Final"


## Imports

import numpy as np
import random as rd
import matplotlib.pyplot as plt
import statistics as stat
import scipy.stats
import time
from scipy.integrate import odeint, quad, simps
import scipy.linalg as lng
import scipy.integrate as integ

# Mesh ploting :
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

start_time = time.time()


## Hyperparameters

GAMMA_DAM = 0.1        # probability of death for damaged cells  (between 0.05 and 0.2)
GAMMA_AD = 0.35       # Probabilite of death for adapted cells  (between 0.2 and 0.5)
DELTA = 0.02     # Probability of repairing for adapted cells

ALPHA_MAX = 0.5    # Maximum of repair probability
SIGMA = 0.5        # Variance of repair curve
MEAN_A = 1        # Mean of repair curve

BETA_MAX = 0.5    # Maximum of adaptation probability

D = 0.3


if __name__ == "__main__" :
    print("___## Version 0.1 ##___\n\n")
    print("___Cell population model for adaptation___\n")

    print(" GAMMA_DAM = ",GAMMA_DAM,"\n","GAMMA_AD = ",GAMMA_AD,"\n",
               "DELTA = ",DELTA,"\n","\n")
    print("Parameters for alpha : \n","ALPHA_MAX = ",ALPHA_MAX," ; SIGMA = ",SIGMA," ; MEAN_A = ",MEAN_A)
    print("Parameters for beta : \n","BETA_MAX = ",BETA_MAX)
    print("Damage coefficient : \n","D = ",D,"\n\n")



## Display functions

def display_1d(x, y, xlab=None, ylab=None, title=None) :
    """
    Plot y versus x as a line with appropriate legend depending on the option.
    """
    fig, ax = plt.subplots()
        
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.setp(ax.spines.values(), linewidth=3)
        
    ax.plot(x,y, linewidth=3)
    
    ax.xaxis.set_tick_params(width=3)
    ax.yaxis.set_tick_params(width=3)
    if xlab != None :
        ax.set_xlabel(xlab, fontsize=20)
    
    if ylab != None :
        ax.set_ylabel(ylab, fontsize=20)
        
    if title != None :
            ax.set_title(title)
    
    fig.show()


def display_2d(x, y, vz, option, zaxis_range=None, xlab=None, ylab=None, title=None) :
    """
    Two-dimensional data plot of vz versus an x,y grid.
    Option should be 1 (colormap plot), 2 (surface plot in 3D) or 3 (both previous plots).
    
    Args:
        x (numpy.ndarray): abscissa discretisation
        y (numpy.ndarray): ordinate discretisation
        vz (numpy.ndarray): matrix of values to plot.
        option (int): choice of graph style.
            1 : 2d colormap graph.
            2 : 3d surface graph.
        zaxis_range (tuple, optional): bounds for the z axis, defaults to None.
    """
    vx, vy = np.meshgrid(x, y)
    if option == 1 :
        fig, ax = plt.subplots()
        c = ax.pcolor(vx, vy, vz,  cmap=cm.coolwarm)
        
        # Labels
        if xlab == None :
            ax.set_xlabel('value of $\mu_b$', fontsize=20)
        else :
            ax.set_xlabel(xlab, fontsize=20) 
        if ylab == None :
            ax.set_ylabel('value of p', fontsize=20)
        else :
            ax.set_ylabel(ylab, fontsize=20)
        
        fig.colorbar(c, ax=ax)
    
    elif option == 2 :
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        surf = ax.plot_surface(vx, vy, vz, cmap=cm.coolwarm,
                                linewidth=0, antialiased=False)
        
        # Customize the z axis.
        if zaxis_range != None :
            ax.set_zlim(zaxis_range[0], zaxis_range[1])
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        
        # Labels
        if xlab == None :
            ax.set_xlabel('value of $\mu_b$', fontsize=20)
        else :
            ax.set_xlabel(xlab, fontsize=20) 
        if ylab == None :
            ax.set_ylabel('value of p', fontsize=20)
        else :
            ax.set_ylabel(ylab, fontsize=20)
        
        
        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)
    
    #title
    if title != None :
        ax.set_title(title) 
    
    fig.show()


## General functions


def laplacian_matrix(n_x) :
    """
    Produce the matrix of the laplacian with Neumann boundary conditions 
    
    Args:
        n_x (int): number of points of the discretisation
    
    Returns:
        numpy.ndarray: matrix of size (n_x,n_x)
    """
    m = np.zeros((n_x,n_x))
    m[0,0] = -1.0
    m[0,1] = 1.0
    for i in range ( 1, n_x -1):
        m[i,i-1] =  1
        m[i,i  ] =  -2.0
        m[i,i+1] =  1
    m[-1,-1] = -1.0
    m[-1,-2] = 1.0
    
    return m

def total_pop(n1, n2, X) :
    N = integ.simps( n1 + n2 , X )
    return N

def source_1(n1, n2, N, r1, delta1) :
    """
    Compute the zero order term in the equation for n1
    
    Args:
        n1 (numpy.ndarray): current repartition of n1 population
        n2 (numpy.ndarray): current repartition of the adapted cells population
        N (float): total mass of the population, pre-computed via total_pop()
        r1 (numpy.ndarray): fitness function for n1
        delta1 (numpy.ndarray): conversion rate from n2 to n1
        
    Returns:
        numpy.ndarray: vector of the zero order source term
    """
    return n1*(r1-N) + delta1*n2

def source_2(n1, n2, N, r2, delta2) :
    """
    Compute the zero order term in the equation for n2
    
    Args:
        n1 (numpy.ndarray): current repartition of n1 population
        n2 (numpy.ndarray): current repartition of the adapted cells population
        N (float): total mass of the population, pre-computed via total_pop()
        r2 (numpy.ndarray): fitness function for n2
        delta2 (numpy.ndarray): conversion rate from n1 to n2
        
    Returns:
        numpy.ndarray: vector of the zero order source term
    """
    return n2*(r2-N) + delta2*n1

def corrective_term(r1, r2, delta1, delta2):
    """
    
    """
    return ( r1 - r2 + np.sqrt( (r1 - r2)**2 + 4*delta1*delta2 ) ) / ( 2 * delta2 )

def generalised_corrective_term(u, dx, r1, r2, delta1, delta2, d=(1,1)):
    """
    
    """
    partial_u=u.copy()
    for i in range(u.size-1):
        partial_u[i] = ( u[i+1] - u[i] ) / dx
    partial_u[-1]= ( u[-1] - u[-2] ) / dx
    
    return ( (d[0]-d[1])*partial_u**2 + r1 - r2 + np.sqrt( r1 - r2 + 4*delta1*delta2 ) ) / ( 2 * delta2 )


def hamiltonian_fitness(q, r1, r2, delta1, delta2):
    """
    
    """
    return (r1+r2)/2 + np.sqrt(  (r1-r2)**2/4 + delta1*delta2  )



## Functions for the "adaptation to DNA damage" model

# Basic functions :

def alpha(s, mean_a=MEAN_A) :
    """
    Compute the probability for a cell to repair DNA damage.
    
    Args:
        t (float): time since damage occured
        mean_a (float): center of the gaussian repair curve
    
    Returns:
        float: probability of successful repair
    """
    return ALPHA_MAX*np.exp(  - (s - mean_a )**2/SIGMA   )


def beta(s,x,p=3) :
    """
    Compute the probability for a cell to adapt to DNA damage.
    
    Args:
        t (float): time since damage occured
        x (float): center of the logistic adaptation curve
        p (float): slope parameter of the logistic adaptation curve.
    
    Returns:
        float: probability of adaptation
    """
    return BETA_MAX/(1+np.exp(-p*( s - x ) ))


# Evolution of the adaptation timing

def repair_weight(s, x) :
    """
    Function to be integrated in repair_vector().
    """
    return alpha(s) * np.exp( -GAMMA_DAM*s - integ.quad(lambda s: alpha(s),0,s)[0]  - integ.quad(lambda s: beta(s,x),0,s)[0] )

def adapt_weight(s, x) :
    """
    Function to be integrated in adapt_vector().
    """
    return beta(s,x) * np.exp( -GAMMA_DAM*s - integ.quad(lambda s: alpha(s),0,s)[0]  - integ.quad(lambda s: beta(s,x),0,s)[0]  )

def repair_vector(X) :
    """
    Pre-computation for the repaired cells flux.
    """
    repair = np.zeros(X.size)
    for i in range(repair.size):
        repair[i] = integ.quad(lambda s: repair_weight(s,X[i]), 0, np.inf)[0]
    return repair

def adapt_vector(X) :
    """
    Pre-computation for the adapted cells flux.
    """
    adapt = np.zeros(X.size)
    for i in range(adapt.size):
        adapt[i] = integ.quad(lambda s: adapt_weight(s,X[i]), 0, np.inf)[0]
    return adapt


def r1_adaptation(X) :
    """
    Fitness for healthy cells, includes the repair term.
    """
    return 1 - D + D*repair_vector(X)

def r2_adaptation(X):
    """
    Fitness for adapted cells, includes conversion to healthy cells.
    """
    return 1 - GAMMA_AD - DELTA 
    

def delta1_adaptation(X):
    """
    Conversion from adapted to healthy.
    """
    return DELTA


def delta2_adaptation(X):
    """
    Conversion from healthy to adapted.
    """
    return D*adapt_vector(X)

##  Evolution of the adaptation heterogeneity 

def repair_weight_p(s, p, x=2) :
    """
    Function to be integrated in repair_vector().
    """
    return alpha(s) * np.exp( -GAMMA_DAM*s - integ.quad(lambda s: alpha(s),0,s)[0]  - integ.quad(lambda s: beta(s,x,p),0,s)[0] )

def adapt_weight_p(s, p, x=2) :
    """
    Function to be integrated in adapt_vector().
    """
    return beta(s,x,p) * np.exp( -GAMMA_DAM*s - integ.quad(lambda s: alpha(s),0,s)[0]  - integ.quad(lambda s: beta(s,x,p),0,s)[0]  )

def repair_vector_p(P, x=2) :
    """
    Pre-computation for the repaired cells flux.
    """
    repair = np.zeros(P.size)
    for i in range(repair.size):
        repair[i] = integ.quad(lambda s: repair_weight_p(s,P[i],x), 0, np.inf)[0]
    return repair

def adapt_vector_p(P, x=2) :
    """
    Pre-computation for the adapted cells flux.
    """
    adapt = np.zeros(P.size)
    for i in range(adapt.size):
        adapt[i] = integ.quad(lambda s: adapt_weight_p(s,P[i],x), 0, np.inf)[0]
    return adapt


def r1_adaptation_p(P, x=2) :
    """
    Fitness for healthy cells, includes the repair term.
    """
    return 1 - D + D*repair_vector_p(P, x)

def r2_adaptation_p(P, x=2):
    """
    Fitness for adapted cells, includes conversion to healthy cells.
    """
    return 1 - GAMMA_AD - DELTA 
    

def delta1_adaptation_p(P, x=2):
    """
    Conversion from adapted to healthy.
    """
    return DELTA


def delta2_adaptation_p(P, x=2):
    """
    Conversion from healthy to adapted.
    """
    return D*adapt_vector_p(P, x)
    

## Simulation of the model


def simulation(n1_0, n2_0, r1, r2, delta1, delta2, t_min=0, t_max=1, dt=0.01, x_min=0, x_max=8, dx=0.1, eps = 0.02, d=(1,1), display_text=False) :
    """
    Compute a numerical simulation of the reduced model up to time t_max.
    The heat equation is solved using a Cranck-Nickolson scheme. Zero-order terms are treated explicitely.
    
    Args:
        n1_0 (numpy.ndarray): initial repartition of n1.
        n2_0 (numpy.ndarray): initial repartition of n2
        r1 (numpy.ndarray): pre-computed r1 vector
        r2 (numpy.ndarray): pre-computed r2 vector
        delta1 (numpy.ndarray): pre-computed delta1 vector
        delta2 (numpy.ndarray): pre-computed delta2 vector
        t_min (float, optional): starting time of the simulation, default to 0.
        t_max (float, optional): final time of the simulation, default to 1.
        dt (float, optional): time step of the simulation, default to 0.01 .
        x_min (float, optional): minimal value for space, default to 0.
        x_max (float, optional): maximal value for space, defaults to 8.
        dx (float, optional): space step of the simulation, defaults to 0.1 .
        eps (float, optional): diffusion coefficient, defaults to 0.02.
        d (tuple, optional): two values for the diffusion parameters d1 and d2, defaults to (1,1)
        display_text (bool, optional): wether to display information on parameters, defaults to False.
    
    Returns:
        numpy.ndarray: final repartition of n1.
        numpy.ndarray: final repartition of n2.
    """
    
    # Time and space discretisation
    n_t = int((t_max+dt-t_min)/dt)
    X=np.arange(0,x_max+dx,dx)
    n_x = X.size
    
    # Initialisation
    n1 = n1_0
    n2 = n2_0
    
    lm = laplacian_matrix(n_x)
    
    scheme_impl_n1 = np.eye(n_x) - (1/2) * d[0] * eps * dt/dx**2 * lm 
    scheme_expl_n1 = np.eye(n_x) + (1/2) * d[0] * eps * dt/dx**2 * lm 
    
    scheme_impl_n2 = np.eye(n_x) - (1/2) * d[1] * eps * dt/dx**2 * lm 
    scheme_expl_n2 = np.eye(n_x) + (1/2) * d[1] * eps * dt/dx**2 * lm 
    
    r2 = 1 - GAMMA_AD - DELTA
    delta1 = DELTA
    
    # Computation of the solution
    for i in range(n_t) :
        
        N = total_pop(n1, n2, X)
        
        n1_tmp = np.dot(scheme_expl_n1, n1) + (1/eps)*dt*source_1(n1, n2, N, r1, delta1)
        n2_tmp = np.dot(scheme_expl_n2, n2) + (1/eps)*dt*source_2(n1, n2, N, r2, delta2)
        n1 = lng.solve( scheme_impl_n1, n1_tmp)
        n2 = lng.solve( scheme_impl_n2, n2_tmp)
    
    
    # Display information about the simulation
    if display_text :
        print("Time from ", t_min, "to ", t_max, "with dt = ", dt)
        print("Space from ", x_min, "to ", x_max, "with dx = ", dx)
        print("Diffusion eps = ", eps)
        print("Initial condition n2_0 = ", n2_0)
        
    return n1, n2   



## Changing environment 


def env1(t):
    return 0.5 * ( 1 + np.cos(4*np.pi*t))



def changing_environment(n1_0, n2_0, env, t_min=0, t_max=1, dt=0.01, x_min=0, x_max=8, dx=0.1, eps = 0.02, d=(1,1), var=("slope",1.5), display_text=False) :
    """
    Compute a numerical simulation of the reduced model up to time t_max.
    The heat equation is solved using a Cranck-Nickolson scheme. Zero-order terms are treated explicitely.
    
    Args:
        n1_0 (numpy.ndarray): initial repartition of n1.
        n2_0 (numpy.ndarray): initial repartition of n2.
        env (function): function giving the environmental variation.
        t_min (float, optional): starting time of the simulation, default to 0.
        t_max (float, optional): final time of the simulation, default to 1.
        dt (float, optional): time step of the simulation, default to 0.01 .
        x_min (float, optional): minimal value for space, default to 0.
        x_max (float, optional): maximal value for space, defaults to 8.
        dx (float, optional): space step of the simulation, defaults to 0.1 .
        eps (float, optional): diffusion coefficient, defaults to 0.02.
        d (tuple, optional): two values for the diffusion parameters d1 and d2, defaults to (1,1)
        display_text (bool, optional): wether to display information on parameters, defaults to False.
    
    Returns:
        numpy.ndarray: final repartition of n1.
        numpy.ndarray: final repartition of n2.
    """
    
    # Time and space discretisation
    n_t = int((t_max+dt-t_min)/dt)
    X=np.arange(0,x_max+dx,dx)
    n_x = X.size
    
    # Initialisation
    n1 = n1_0
    n2 = n2_0
    
    lm = laplacian_matrix(n_x)
    
    scheme_impl_n1 = np.eye(n_x) - (1/2) * d[0] * eps * dt/dx**2 * lm 
    scheme_expl_n1 = np.eye(n_x) + (1/2) * d[0] * eps * dt/dx**2 * lm 
    
    scheme_impl_n2 = np.eye(n_x) - (1/2) * d[1] * eps * dt/dx**2 * lm 
    scheme_expl_n2 = np.eye(n_x) + (1/2) * d[1] * eps * dt/dx**2 * lm 
    
    print("n_t = ",n_t)
    
    # Computation of the solution
    for i in range(n_t) :
        
        print(i)
        
        repair = np.zeros(X.size)
        if var[0] == "mean":
            for i in range(repair.size):
                repair[i] = integ.quad(lambda s: env(t_min + i*dt)*alpha(s) * np.exp( -GAMMA_DAM*s - env(t_min + i*dt)*integ.quad(lambda s: alpha(s),0,s)[0]  - integ.quad(lambda s: beta(s,X[i],var[1]),0,s)[0] ), 0, np.inf)[0]
        elif var[0] == "slope":
            for i in range(repair.size):
                repair[i] = integ.quad(lambda s: env(t_min + i*dt)*alpha(s) * np.exp( -GAMMA_DAM*s - env(t_min + i*dt)*integ.quad(lambda s: alpha(s),0,s)[0]  - integ.quad(lambda s: beta(s,var[1],X[i]),0,s)[0] ), 0, np.inf)[0]
        
        adapt = np.zeros(X.size)
        if var[0] == "mean":
            for i in range(adapt.size):
                adapt[i] = integ.quad(lambda s:  beta(s,X[i],var[1]) * np.exp( -GAMMA_DAM*s - env(t_min + i*dt)*integ.quad(lambda s: alpha(s),0,s)[0]  - integ.quad(lambda s: beta(s,x[i],var[1]),0,s)[0]), 0, np.inf)[0]
        elif var[0] == "slope":
            for i in range(adapt.size):
                adapt[i] = integ.quad(lambda s: beta(s,var[1],X[i]) * np.exp( -GAMMA_DAM*s - env(t_min + i*dt)*integ.quad(lambda s: alpha(s),0,s)[0]  - integ.quad(lambda s: beta(s,var[1],X[i]),0,s)[0]), 0, np.inf)[0]
        
        r1 = 1 - D + D*repair
        delta2 = D*adapt
        
        N = total_pop(n1, n2, X)
        
        n1_tmp = np.dot(scheme_expl_n1, n1) + (1/eps)*dt*source_1(n1, n2, N, r1, delta1)
        n2_tmp = np.dot(scheme_expl_n2, n2) + (1/eps)*dt*source_2(n1, n2, N, r2, delta2)
        n1 = lng.solve( scheme_impl_n1, n1_tmp)
        n2 = lng.solve( scheme_impl_n2, n2_tmp)
        
    
    # Display information about the simulation
    if display_text :
        print("Time from ", t_min, "to ", t_max, "with dt = ", dt)
        print("Space from ", x_min, "to ", x_max, "with dx = ", dx)
        print("Diffusion eps = ", eps)
        print("Initial condition n2_0 = ", n2_0)
        
    return n1, n2   




## Tests

# 
# # Time:
# t_min = 0
# t_max = 110
# dt = 0.01
# 
# # Space:
# x_min = 0
# x_max = 12
# dx = 0.05
# X = np.arange(0,x_max+dx,dx)
# 
# # Scale :
# eps = 0.001
# 
# # Initial condition :
# # n_0 = 1/8 * np.ones(X.size)
# 
# n_0 = X.copy()
# for i in range(n_0.size) :
#     n_0[i] = 4  *np.exp(-10*(X[i]-4)**2)
# 
# # n_0 = 1/8 * np.ones(X.size)
# 
# # Précalcul
# repair = repair_vector(X)
# adapt = adapt_vector(X)
# 
# # Initialisation
# n, a = reduced_problem(n_0, None, t_min, t_max, dt, x_min, x_max, dx,
#                                                      eps, repair, adapt, False)
# 
# # Itération
# fig, ax = plt.subplots()
# # ax.plot(X,n_0)
# for i in range(30) :
#     n, a = reduced_problem(n, a, t_min, t_max, dt, x_min, x_max, dx,
#                                                      eps, repair, adapt, False, False)
#     
# ax.plot(X,n, label="eps=0.001")
# 
# 
# 
# eps = 0.0001
# n_0 = X.copy()
# for i in range(n_0.size) :
#     n_0[i] = 4  *np.exp(-10*(X[i]-4)**2)
# for i in range(30) :
#     n, a = reduced_problem(n, a, t_min, t_max, dt, x_min, x_max, dx,
#                                                      eps, repair, adapt, False, False)    
# ax.plot(X,n, label="eps=0.0001")
# 
#  
# ax.legend()
# 
# fig.show()

##  Numerical tests



if __name__ == "__main__" :
    
    
    
    # Figure 1 : equivalent fitness r(x) with respect to adaptation timing x
    
    x_min = 0
    x_max = 10
    dx = 0.05
    X = np.arange(0,x_max+dx,dx)
    
    r1 = r1_adaptation(X)
    r2 = r2_adaptation(X)
    delta1 = delta1_adaptation(X)
    delta2 = delta2_adaptation(X)
        
    q = corrective_term(r1, r2, delta1, delta2)
    r_H = hamiltonian_fitness(q, r1, r2, delta1, delta2)
        
    display_1d(X,r_H,"Value of the timing parameter x", "Hamiltonian fitness $r_H(x)$" )

    
    # Figure 2a : equivalent fitness r(p) with respect to adaptation heterogeneity p
    
    p_min = 0.01
    p_max = 10
    dp = 0.05
    P = np.arange(0,p_max+dp,dp)
    
    r1 = r1_adaptation_p(P)
    r2 = r2_adaptation_p(P)
    delta1 = delta1_adaptation_p(P)
    delta2 = delta2_adaptation_p(P)
        
    q = corrective_term(r1, r2, delta1, delta2)
    r_H = hamiltonian_fitness(q, r1, r2, delta1, delta2)
    
    display_1d(P,r_H,"Value of the heterogeneity parameter p", "Hamiltonian fitness $r_H(p)$" )
    
    
    # Figure 2b : equivalent fitness r(p) with respect to adaptation heterogeneity p
    
    p_min = 0.01
    p_max = 10
    dp = 0.05
    P = np.arange(0,p_max+dp,dp)
    
    r1 = r1_adaptation_p(P,20)
    r2 = r2_adaptation_p(P,20)
    delta1 = delta1_adaptation_p(P,20)
    delta2 = delta2_adaptation_p(P,20)
        
    q = corrective_term(r1, r2, delta1, delta2)
    r_H = hamiltonian_fitness(q, r1, r2, delta1, delta2)
    
    display_1d(P,r_H,"Value of the heterogeneity parameter p", "Hamiltonian fitness $r_H(p)$" )
    
    
    
    # 
    # # Figure 3
    # 
    # # Time:
    # t_min = 0
    # t_max = 100
    # dt = 0.05
    # 
    # # Space:
    # p_min = 0.03
    # p_max = 5
    # dp = 0.05
    # P = np.arange(0,p_max+dp,dp)
    # 
    # # Scale :
    # eps = 0.1
    # 
    # # Initial condition :
    # # n_0 = 1/8 * np.ones(X.size)
    # 
    # n1_0 = P.copy()
    # for i in range(n1_0.size) :
    #     n1_0[i] =   0.2*np.exp(-10*(P[i]-2)**2)
    # 
    # n2_0 = np.zeros(P.size)
    # 
    # r1 = r1_adaptation_p(P)
    # r2 = r2_adaptation_p(P)
    # delta1 = delta1_adaptation_p(P)
    # delta2 = delta2_adaptation_p(P)
    # print("hell yeah!")
    # 
    # n1, n2 = simulation(n1_0, n2_0, r1, r2, delta1, delta2, t_min, t_max, dt, p_min, p_max, dp, eps, d=(1,1), display_text=False)
    # 
    # display_1d(P,n1,"Value of heterogeneity parameter p", "Density of the population" )
    
    
    
    # # Figure 4
    # 
    # 
    # # Time:
    # t_min = 0
    # t_max = 10
    # dt = 0.05
    # 
    # # Space:
    # p_min = 0.03
    # p_max = 5
    # dp = 0.05
    # P = np.arange(0,p_max+dp,dp)
    # 
    # # Scale :
    # eps = 0.1
    # 
    # # Initial condition :
    # # n_0 = 1/8 * np.ones(X.size)
    # 
    # n1_0 = P.copy()
    # for i in range(n1_0.size) :
    #     n1_0[i] =   0.2*np.exp(-10*(P[i]-2)**2)
    # 
    # n2_0 = np.zeros(P.size)
    # 
    # n1, n2 = changing_environment(n1_0, n2_0, lambda t: env1(t/eps), t_min, t_max, dt, p_min, p_max, dp, eps, (1,1), ("slope",1.5)) 
    # 
    # display_1d(P,n1,"Value of heterogeneity parameter p", "Density of the population" )
    # 
    
    # Figure 5



    # 
    # 
    # # Time:
    # t_min = 0
    # t_max = 10
    # dt = 0.0001
    # 
    # # Space:
    # x_min = 0
    # x_max = 8
    # dx = 0.05
    # X = np.arange(0,x_max+dx,dx)
    # 
    # # Scale :
    # eps = 0.001
    # 
    # # Initial condition :
    # # n_0 = 1/8 * np.ones(X.size)
    # 
    # n1_0 = X.copy()
    # for i in range(n1_0.size) :
    #     n1_0[i] =   0.2*np.exp(-10*(X[i]-4)**2)
    # 
    # n2_0 = np.zeros(X.size)
    # 
    # r1 = r1_adaptation(X)
    # r2 = r2_adaptation(X)
    # delta1 = delta1_adaptation(X)
    # delta2 = delta2_adaptation(X)
    # print("hell yeah!")
    # 
    # # Corrective term and all
    # 
    # q = corrective_term(r1, r2, delta1, delta2)
    # r = equivalent_fitness(q, r1, r2, delta1, delta2)
    # 
    # fig, ax = plt.subplots()
    # ax.plot(X,r)
    # fig.show()
    # 
    # 
    # # Initialisation
    # n1, n2 = simulation(n1_0, n2_0, r1, r2, delta1, delta2, t_min, t_max, dt, x_min, x_max, dx, eps, d=(1,1), display_text=False)
    # 
    # 
    # # Further simulation
    # n1, n2 = simulation(n1, n2, r1, r2, delta1, delta2, t_min, t_max, dt, x_min, x_max, dx, eps, d=(1,1), display_text=False)
    # 
    # 
    # # plot
    # fig, ax = plt.subplots()
    # ax.plot(X,n1_0, label="initial")
    # ax.plot(X,n1, label="final")
    # ax.legend()
    # fig.show()
    # 


## Execution time

end_time = time.time()

if __name__ == "__main__" :
    print("Execution time : ",end_time-start_time)
