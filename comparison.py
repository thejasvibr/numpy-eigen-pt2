#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comparing the vector based Numpy and Eigen C++ implementations
==============================================================
Created on Mon Sep 26 15:18:17 2022

@author: thejasvi
"""
import glob
import numpy as np 
import time
try:
    import os 
    os.environ['EXTRA_CLING_ARGS'] = '-fopenmp'
    #pch_path = os.getcwd()
    #import cppyy_backend.loader as l
    #l.set_cling_compile_options(True)
    #l.ensure_precompiled_header(pch_path)
    #full_path = glob.glob(pch_path+'/allD*')[0]
    #os.environ['CLING_STANDARD_PCH'] = full_path
    import cppyy
    #cppyy.load_library('/home/thejasvi/anaconda3/lib/libiomp5.so')
    cppyy.load_library('/usr/lib/llvm-9/lib/libiomp5.so')
    #cppyy.load_library('/home/autumn/anaconda3/lib/libiomp5.so')
    cppyy.add_include_path('../np_vs_eigen/eigen')
    cppyy.include('sw2002_vectorbased.cpp')
except ImportError:
    pass

from sw2002_vectorbased_pll import sw_matrix_optim
# make neat wrapper. 

def cppyy_sw2002(micntde, nmics):
    as_Vxd = cppyy.gbl.sw_matrix_optim(cppyy.gbl.std.vector[float](micntde.tolist()),
                                       nmics)
    return np.array(as_Vxd, dtype=np.float64)
    
uu = np.array([0.1, 0.6, 0.9,
 			3.61, 54.1, 51.1,
 			68.1, 7.1,  8.1,
 			9.1,  158.1, 117.1,
 			18.1, 99.1, 123.1,
 			12.1, 13.1, 14.1, 19.1], dtype=np.float64)
uu[-4:] *= 1e-3
nruns = 500000
#np.random.seed(82319)
nmics = 10
ncols = nmics*3 + nmics-1
many_u = np.random.normal(0,1,ncols*nruns).reshape(nruns,ncols)
many_u[:,-(nmics-1):] *= 1e-3
many_u[:,:-(nmics-1)] += np.random.normal(.1,0.5,(ncols-(nmics-1))*nruns).reshape(nruns,ncols-(nmics-1))
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
print(f'Overall time taken Eigen: {stop-start} s')

#%%

start = time.perf_counter_ns()/1e9
for i in range(nruns):
    all_solutions_numpy[i,:] = sw_matrix_optim(many_u[i,:], 5)
stop = time.perf_counter_ns()/1e9
avg_numpy = (stop-start)/nruns
print(f'time taken for NumPy {nruns} runs: {avg_numpy*1e6} micro s ')
print(f'Overall time taken NumPy: {stop-start} s')

import scipy.spatial as spl
discrepancy = np.zeros((nruns,2))
# calculate largest distance between predicted points
def calc_solution_distances(X,Y):
    distances = np.zeros(2)
    distances[0] =  spl.distance.euclidean(X[:3], Y[:3])
    distances[1] =  spl.distance.euclidean(X[3:], Y[3:])
    return distances
    
for r in range(nruns):
    discrepancy[r,:] = calc_solution_distances(all_solutions_cpy[r,:],
                                               all_solutions_numpy[r,:])

max_discrepancy = np.max(discrepancy,1)

print(f'Max discrepancy between NumPy and Eigen QR methods: {np.max(max_discrepancy)}')

print(f'\n \n Overall speedup by using Eigen: {avg_numpy/avg_cpy}')
# assert np.allclose(all_solutions_cpy, all_solutions_numpy, atol=1e-2)==True

#%% Check where the correspondence drops. Of course, in one case the np.linalg.pinv
# is using the SVD - a time-intesive method, while the curent Eigen implementation
# uses the QR method - a faster but less stable version. 

nonmatching = np.argwhere(np.abs(all_solutions_cpy-all_solutions_numpy)>1e-3)
nonmatching_rows = np.unique(nonmatching[:,0])


#%% Here let's also run the pll version. 

def pll_cppyy_sw2002(many_micntde, many_nmics, num_cores, c):
    block_in = cppyy.gbl.std.vector[cppyy.gbl.std.vector[float]](many_micntde.shape[0])
    block_mics = cppyy.gbl.std.vector[int](many_micntde.shape[0])
    
    for i in range(many_micntde.shape[0]):
        block_in[i] = cppyy.gbl.std.vector[float](many_micntde[i,:].tolist())
        block_mics[i] = int(many_nmics[i])
    block_out = cppyy.gbl.pll_sw_optim(block_in, block_mics, num_cores, c)
    return block_out

many_mcs = np.tile(nmics, many_u.shape[0]).tolist()
ncores = os.cpu_count()
st = time.perf_counter_ns()
uu = pll_cppyy_sw2002(many_u, many_mcs, ncores, 343.0)
stop = time.perf_counter_ns()
pll_durn = (stop-st)/1e9
print(f'OMP pll version takes: {pll_durn} s')
print(f'Pll vs Serial speedup : {avg_cpy/(pll_durn/nruns)}')

    

