#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 22:37:17 2020

@author: marios
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
from scipy.integrate import odeint
   
g = 9.81 # m/s
# For m_1 = m_2 = l_1 = l_2 = 1
#H = ((p_theta1**2 + 2*p_theta2**2 - 2*p_theta1*p_theta2*np.cos(theta_1 - theta_2)) / (2 + 2*(np.sin(theta_1 - theta_2))**2)) \
#                                - 2*g*(np.cos(theta_1) + 0.5*np.cos(theta_2))
    
#dttheta_1 = (p_theta1 - p_theta2 * np.cos(theta_1 - theta_2)) / (1+(np.sin(theta_1 - theta_2))**2)
#dttheta_2 = (2*p_theta2 - p_theta1 * np.cos(theta_1 - theta_2)) / (1+(np.sin(theta_1 - theta_2))**2)

#dtp_theta1 = -2*g*np.sin(theta_1) - ((p_theta1*p_theta_2*np.sin(theta_1 - theta_2)) / (1 + (np.sin(theta_1 - theta_2))**2)) \
#+ np.sin(2*(theta_1 - theta_2))*((p_theta1**2 + 2*p_theta2**2 - 2*p_theta1*p_theta2*np.cos(theta_1 - theta_2)) / (2 + 2*(np.sin(theta_1 - theta_2))**2)**2)

#dtp_theta2 = -g*np.sin(theta_2) + ((p_theta1*p_theta_2*np.sin(theta_1 - theta_2)) / (1 + (np.sin(theta_1 - theta_2))**2)) \
#- np.sin(2*(theta_1 - theta_2))*((p_theta1**2 + 2*p_theta2**2 - 2*p_theta1*p_theta2*np.cos(theta_1 - theta_2)) / (2 + 2*(np.sin(theta_1 - theta_2))**2)**2)


###################
# Symplectic Euler
####################
def symEuler(Ns, theta1_0,theta2_0, p_theta1_0, p_theta2_0, t0, t_max,lam=1):
    t_s = np.linspace(t0, t_max, Ns+1)
    ts = t_max/Ns
    dts = t_max/Ns; 
    
    theta1_s = np.zeros(Ns+1); p_theta1_s = np.zeros(Ns+1);
    theta2_s = np.zeros(Ns+1); p_theta2_s = np.zeros(Ns+1)
     
    theta1_s[0], p_theta1_s[0], theta2_s[0], p_theta2_s[0] = theta1_0, p_theta1_0, theta2_0, p_theta2_0
    for n in range(Ns):
        theta1_s[n+1] = theta1_s[n] + dts*(p_theta1_s[n] - p_theta2_s[n] * np.cos(theta1_s[n] - theta2_s[n])) / (1+(np.sin(theta1_s[n] - theta2_s[n]))**2)
        theta2_s[n+1] = theta2_s[n] + dts*(2*p_theta2_s[n] - p_theta1_s[n] * np.cos(theta1_s[n] - theta2_s[n])) / (1+(np.sin(theta1_s[n] - theta2_s[n]))**2)
        
        p_theta1_s[n+1] = p_theta1_s[n] + dts*(-2*g*np.sin(theta1_s[n]) - ((p_theta1_s[n]*p_theta2_s[n]*np.sin(theta1_s[n] - theta2_s[n])) / (1 + (np.sin(theta1_s[n] - theta2_s[n]))**2)) \
+ np.sin(2*(theta1_s[n] - theta2_s[n]))*((p_theta1_s[n]**2 + 2*p_theta2_s[n]**2 - 2*p_theta1_s[n]*p_theta2_s[n]*np.cos(theta1_s[n] - theta2_s[n])) / (2 + 2*(np.sin(theta1_s[n] - theta2_s[n]))**2)**2))

        p_theta2_s[n+1] = p_theta2_s[n] + dts*(-g*np.sin(theta2_s[n]) + ((p_theta1_s[n]*p_theta2_s[n]*np.sin(theta1_s[n] - theta2_s[n])) / (1 + (np.sin(theta1_s[n] - theta2_s[n]))**2)) \
- np.sin(2*(theta1_s[n] - theta2_s[n]))*((p_theta1_s[n]**2 + 2*p_theta2_s[n]**2 - 2*p_theta1_s[n]*p_theta2_s[n]*np.cos(theta1_s[n] - theta2_s[n])) / (2 + 2*(np.sin(theta1_s[n] - theta2_s[n]))**2)**2))


        # E_euler = energy( x_s, y_s, px_s, py_s, lam)

    E_euler = energy( theta1_s, theta2_s, p_theta1_s, p_theta2_s, lam)
    return E_euler, theta1_s,theta2_s, p_theta1_s, p_theta2_s, t_s
 
   
    
   
# Use below in the Scipy Solver   
def f(u, t,lam=1):
    theta1, theta2, p_theta1, p_theta2 = u      # unpack current values of u
    derivs = [(p_theta1-p_theta2*np.cos(theta1-theta2))/(1+(np.sin(theta1-theta2))**2), (2*p_theta2-p_theta1*np.cos(theta1-theta2))/(1+(np.sin(theta1-theta2))**2), (-2*g*np.sin(theta1) - ((p_theta1*p_theta2*np.sin(theta1 - theta2)) / (1 + (np.sin(theta1- theta2))**2)) \
+ np.sin(2*(theta1 - theta2))*((p_theta1**2 + 2*p_theta2**2 - 2*p_theta1*p_theta2*np.cos(theta1 - theta2)) / (2 + 2*(np.sin(theta1- theta2))**2)**2)), (-g*np.sin(theta2) + ((p_theta1*p_theta2*np.sin(theta1 - theta2)) / (1 + (np.sin(theta1- theta2))**2)) \
- np.sin(2*(theta1 - theta2))*((p_theta1**2 + 2*p_theta2**2 - 2*p_theta1*p_theta2*np.cos(theta1 - theta2)) / (2 + 2*(np.sin(theta1- theta2))**2)**2)) ]     # list of dy/dt=f functions
    return derivs

# Scipy Solver   
def DoublePendulumsolution(N,t, theta1_0, theta2_0, p_theta1_0, p_theta2_0,lam=1):
    u0 = [theta1_0, theta2_0, p_theta1_0, p_theta2_0]
    # Call the ODE solver
    solPend = odeint(f, u0, t)
    theta1_P = solPend[:,0];    theta2_P  = solPend[:,1];
    p_theta1_P = solPend[:,2];   p_theta2_P = solPend[:,3]
    return theta1_P,theta2_P, p_theta1_P, p_theta2_P

# Energy of Double Pendulum system
def energy(theta1, theta2, p_theta1, p_theta2, lam=1):    
    Ntheta1=len(theta1); 
    theta1=theta1.reshape(Ntheta1);      theta2=theta2.reshape(Ntheta1)
    p_theta1=p_theta1.reshape(Ntheta1);    p_theta2=p_theta2.reshape(Ntheta1)

    E = ((p_theta1**2 + 2*p_theta2**2 - 2*p_theta1*p_theta2*np.cos(theta1 - theta2)) / (2 + 2*(np.sin(theta1 - theta2))**2)) \
                                - 2*g*(np.cos(theta1) + 0.5*np.cos(theta2))
    E = E.reshape(Ntheta1)
    return E

# initial energy
def DoublePendulum_exact(N,theta1_0, theta2_0, v_theta1_0, v_theta2_0, lam=1):
    E0 = ((v_theta1_0**2 + 2*v_theta2_0**2 - 2*v_theta1_0*v_theta2_0*np.cos(theta1_0 - theta2_0)) / (2 + 2*(np.sin(theta1_0 - theta2_0))**2)) \
                                - 2*g*(np.cos(theta1_0) + 0.5*np.cos(theta2_0))
    E_ex = E0*np.ones(N);
    return E0, E_ex





def saveData(path, t, theta1, theta2, p_theta1,p_theta2, E):
    np.savetxt(path+"t.txt",t)
    np.savetxt(path+"theta1.txt",theta1)
    np.savetxt(path+"theta2.txt",theta2)
    np.savetxt(path+"p_theta1.txt",p_theta1)
    np.savetxt(path+"p_theta2.txt",p_theta2)
    np.savetxt(path+"E.txt",E)

    
    