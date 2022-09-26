#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparing the vector based Numpy and Eigen C++ implementations
==============================================================
Created on Mon Sep 26 15:18:17 2022

@author: thejasvi
"""
import numpy as np 
import time
try:
    import cppyy
    cppyy.add_include_path('../np_vs_eigen/eigen')
    cppyy.include('sw2002_vectorbased.cpp')
except ImportError:
    pass
#%%
from sw2002_vectorbased_pll import sw_matrix_optim
# make neat wrapper. 

def cppyy_sw2002(micntde, nmics):
    as_vectdouble = cppyy.gbl.std.vector[float](micntde.tolist())
    as_Vxd = cppyy.gbl.sw_matrix_optim(as_vectdouble, nmics)
    return np.array(as_Vxd, dtype=np.float64)
    
uu = np.array([0.1, 0.6, 0.9,
 			3.61, 54.1, 51.1,
 			68.1, 7.1,  8.1,
 			9.1,  158.1, 117.1,
 			18.1, 99.1, 123.1,
 			12.1, 13.1, 14.1, 19.1], dtype=np.float64)
uu[-4:] *= 1e-3
nruns = 50000
#np.random.seed(82319)
nmics = 5
ncols = nmics*3 + nmics-1
many_u = np.random.normal(0,1,ncols*nruns).reshape(nruns,ncols)
many_u[:,-(nmics-1):] *= 1e-3

#%%
# 'Warm up' the cppyy function run by calling it once3. 
aa = sw_matrix_optim(uu, 5)
bb = cppyy_sw2002(uu, 5)


#%%
# Warm up the cppyy function by calling it once. 
all_solutions_cpy = np.zeros((nruns, 6))
all_solutions_numpy = np.zeros((nruns, 6))
#%%
print('miaow \n .....')
start = time.perf_counter_ns()/1e9
for i in range(nruns):
    all_solutions_cpy[i,:] = cppyy_sw2002(many_u[i,:], 5)
stop = time.perf_counter_ns()/1e9
avg_cpy = (stop-start)/nruns
print(f'time taken for cppyy {nruns} runs: {avg_cpy*1e6} micro s ')

#%%

start = time.perf_counter_ns()/1e9
for i in range(nruns):
    all_solutions_numpy[i,:] = sw_matrix_optim(many_u[i,:], 5)
stop = time.perf_counter_ns()/1e9
avg_numpy = (stop-start)/nruns
print(f'time taken for NumPy {nruns} runs: {avg_numpy*1e6} micro s ')

assert np.allclose(all_solutions_cpy, all_solutions_numpy, atol=1e-3)==True
print(f'\n \n Overall speedup by using Eigen: {avg_numpy/avg_cpy}')



