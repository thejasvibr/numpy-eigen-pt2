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
from joblib import Parallel, delayed
import os 
import tqdm
compile_start = time.perf_counter_ns()/1e9
os.environ['EXTRA_CLING_ARGS'] = '-fopenmp'
pch_path = os.getcwd()
#import cppyy_backend.loader as l
#l.set_cling_compile_options(True)
#l.ensure_precompiled_header(pch_path)
#full_path = glob.glob(pch_path+'/allD*')[0]
#os.environ['CLING_STANDARD_PCH'] = full_path
import cppyy
#cppyy.load_library('/home/thejasvi/anaconda3/lib/libiomp5.so')
#cppyy.load_library('/usr/lib/llvm-9/lib/libiomp5.so')
cppyy.load_library('C:\\Users\\theja\\anaconda3\\Library\\bin\\libiomp5md.dll')
#cppyy.load_library('/home/autumn/anaconda3/lib/libiomp5.so')
cppyy.add_include_path('../np_vs_eigen/eigen')
cppyy.include('sw2002_vectorbased.cpp')
compile_stop = time.perf_counter_ns()/1e9
print(f'{compile_stop-compile_start} s for compilation')
np.random.seed(8239)
#%%
from sw2002_vectorbased_pll import sw_matrix_optim
# make neat wrapper. 

def cppyy_sw2002(micntde, nmics):
    as_Vxd = cppyy.gbl.sw_matrix_optim(cppyy.gbl.std.vector[float](micntde.tolist()),
                                       nmics)
    return np.array(as_Vxd, dtype=np.float64)

nruns = int(5e3)
np.random.seed(8239)
nmics = 5
ncols = nmics*3 + nmics-1

def make_mock_data(nmics):
    xyz_range = np.linspace(-5,5,1000)
    micxyz = np.random.choice(xyz_range, nmics*3).reshape(-1,3)
    source = np.random.choice(xyz_range, 3)
    mic_to_source_dist = np.apply_along_axis(np.linalg.norm, 1, micxyz-source)
    R_ref0 = mic_to_source_dist[1:] - mic_to_source_dist[0]
    output = np.concatenate((micxyz.flatten(), R_ref0)).flatten()
    return output, source

def make_block_mock_data(nmics, blocksize):
    ncols = nmics*3 + nmics - 1
    outputs = np.zeros((blocksize, ncols))
    sources = np.zeros((blocksize,3))
    for i in tqdm.trange(blocksize):
        outputs[i,:], ss = make_mock_data(nmics)
        sources[i,:] = ss
    return outputs, sources

def pll_make_mock_data(nmics, nruns):
    num_cores = os.cpu_count()
    splits = [len(each) for each in np.array_split(np.arange(nruns), num_cores)]
    outputs = Parallel(n_jobs=num_cores)(delayed(make_block_mock_data)(nmics, blocksize) for blocksize in splits)
    sim_data = np.row_stack([each[0] for each in outputs])
    sources = np.row_stack([each[1] for each in outputs])
    return sim_data, sources

# make mock data
print('Making sim daa')
many_u, sources = pll_make_mock_data(nmics, nruns)
print('Done w sim data')
#%% Parallelise the creationg of mock data.
# many_u = np.zeros((nruns,ncols))
# sources = np.zeros((nruns,3))
# for i in range(nruns):
#     many_u[i,:], thissource = make_mock_data(nmics)
#     sources[i,:] = thissource
ii = cppyy_sw2002(many_u[0,:], nmics)
#%%
# 'Warm up' the cppyy function run by calling it once3. 
# serial_sta = time.perf_counter_ns()/1e9
# for i in range(many_u.shape[0]):
#     ##print(f'{i} index')
#     try:
#         bb = cppyy_sw2002(many_u[i,:], nmics)
#     except:
#         print(f'(Problem at {i})')    
#         break
# serial_stop = time.perf_counter_ns()/1e9
#     #print(aa, bb)
# print('Done with serial fun:', serial_stop-serial_sta)
#%% Here let's also run the pll version. 

def pll_cppyy_sw2002(many_micntde, many_nmics, num_cores, c):
    block_in = cppyy.gbl.std.vector[cppyy.gbl.std.vector[float]](many_micntde.shape[0])
    block_mics = cppyy.gbl.std.vector[int](many_micntde.shape[0])
    
    for i in range(many_micntde.shape[0]):
        block_in[i] = cppyy.gbl.std.vector[float](many_micntde[i,:].tolist())
        block_mics[i] = int(many_nmics[i])
    block_out = cppyy.gbl.pll_sw_optim(block_in, block_mics, num_cores, c)
    return block_out
print('Starting pll cppyy run')
many_mcs = np.tile(nmics, many_u.shape[0]).tolist()
ncores = os.cpu_count()
st = time.perf_counter_ns()
uu = pll_cppyy_sw2002(many_u, many_mcs, ncores, 343.0)
stop = time.perf_counter_ns()
pll_durn = (stop-st)/1e9
print(f'OMP pll version takes: {pll_durn} s')
# print(f'Pll vs Serial speedup : {avg_cpy/(pll_durn/nruns)}')

#%%
from joblib import Parallel, delayed

def block_numpy_sw2002(many_micntde, many_nmics,c):
    rows, _ = many_micntde.shape
    solutions = np.zeros((rows, 3))
    
    for i in range(rows):
        solutions[i,:] = sw_matrix_optim(many_micntde[i,:], many_nmics[i])
    return solutions

def pll_numpy_sw2002(many_micntde, many_nmics, c):
    numcores = os.cpu_count();
    indices = np.array_split(np.arange(many_micntde.shape[0]), numcores)
    blocks_micntde = [many_micntde[block,:] for block in indices]
    blocks_nmics = [many_nmics[block] for block in indices]
    outputs = Parallel(n_jobs=numcores)(delayed(block_numpy_sw2002)(tde_block, nmic_block,c) for (tde_block, nmic_block) in zip(blocks_micntde, blocks_nmics))
    all_out = np.row_stack(outputs)
    return all_out

print('Starting Numpy pll run')
sta = time.perf_counter_ns()/1e9
np_pll = pll_numpy_sw2002(many_u, np.tile(nmics, nruns), 343.0)
sto = time.perf_counter_ns()/1e9
pll_np = sto-sta
print(f'Python Pll total durn: {sto-sta} s')


#%%
print(f' OMP C++ vs Joblib Python: {pll_durn , pll_np}')
print(f'C++ advantage is: {pll_np/pll_durn}')
    
#%% Now check the accuracy of outputs
cpy_error = np.array([np.linalg.norm(sources[each,:]-np.array(uu[each])) for each in range(nruns)])
np_error = np.array([np.linalg.norm(sources[each,:]-np_pll[each,:]) for each in range(nruns)])
