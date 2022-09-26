#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementing a vector based implementation of SW2002
====================================================
Created on Mon Sep 26 13:40:03 2022

@author: thejasvi
"""

import numpy as np
#np.random.seed(82319) 
matmul = np.matmul

    
def sw_matrix_optim(mic_ntde_orig, nmics, c=343.0):
    '''
    mic_ntde : 3*nmics + nmics-1
    '''
    mic_ntde = mic_ntde_orig.copy()
    #print(mic_ntde.shape)
    position_inds = nmics*3
    mic0 = mic_ntde[:3]
    starts, stops = np.arange(3,position_inds,3), np.arange(6,position_inds+3,3)
    for start, stop in zip(starts, stops):
        mic_ntde[start:stop] -= mic0
    
    tau = mic_ntde[-(nmics-1):]/c
    R = mic_ntde[3:position_inds].reshape(-1,3)
    
    R_inv = np.linalg.pinv(R)

    Nrec_minus1 = R.shape[0]
    b = np.zeros(Nrec_minus1)
    f = np.zeros(Nrec_minus1)
    g = np.zeros(Nrec_minus1)
    #print(R, tau)
    for i in range(Nrec_minus1):
        b[i] = np.linalg.norm(R[i,:])**2 - (c*tau[i])**2
        f[i] = (c**2)*tau[i]
        g[i] = 0.5*(c**2-c**2)

    a1 = matmul(matmul(R_inv, b).T, matmul(R_inv,b))
    a2 = matmul(matmul(R_inv, b).T, matmul(R_inv,f))
    a3 = matmul(matmul(R_inv, f).T, matmul(R_inv,f))
    

    a_quad = a3 - c**2
    b_quad = -a2
    c_quad = a1/4.0

    t_soln1 = (-b_quad + np.sqrt(b_quad**2 - 4*a_quad*c_quad))/(2*a_quad)
    t_soln2 = (-b_quad - np.sqrt(b_quad**2 - 4*a_quad*c_quad))/(2*a_quad)

    s12 = np.zeros(6)
    s12[:3] = matmul(R_inv,b*0.5) - matmul(R_inv,f)*t_soln1
    s12[3:] = matmul(R_inv,b*0.5) - matmul(R_inv,f)*t_soln2

    s12[:3] += mic0
    s12[3:] += mic0
    return s12