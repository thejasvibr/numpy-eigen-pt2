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
from joblib import Parallel, delayed
import os 
import tqdm
compile_start = time.perf_counter_ns()/1e9
os.environ['EXTRA_CLING_ARGS'] = '-fopenmp -O2'
# pch_path = os.getcwd()
# import cppyy_backend.loader as l
# l.set_cling_compile_options(True)
# #l.ensure_precompiled_header(pch_path)
# full_path = glob.glob(pch_path+'/allD*')[0]
# os.environ['CLING_STANDARD_PCH'] = full_path
import cppyy
#cppyy.load_library('/home/thejasvi/anaconda3/lib/libiomp5.so')
#cppyy.load_library('/usr/lib/llvm-9/lib/libiomp5.so')
cppyy.load_library('C:\\Users\\theja\\anaconda3\\Library\\bin\\libiomp5md.dll')
#cppyy.load_library('/home/autumn/anaconda3/lib/libiomp5.so')
cppyy.add_include_path('../np_vs_eigen/eigen')
cppyy.include('sw2002_vectorbased.cpp')
compile_stop = time.perf_counter_ns()/1e9
print(f'{compile_stop-compile_start} s for compilation')


from sw2002_vectorbased_pll import sw_matrix_optim as swo_py
# make neat wrapper. 
#%%
def cppyy_sw2002(micntde, nmics):
    as_Vxd = cppyy.gbl.sw_matrix_optim(cppyy.gbl.std.vector['double'](micntde.tolist()),
                                       nmics)
    return np.array(as_Vxd, dtype=np.float64)

nruns = int(1e6)
np.random.seed(569)
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

parallel = True
# make mock data
print('Making sim daa')
if not parallel:
    many_u, sources = make_block_mock_data(nmics, nruns)
else:
    many_u, sources = pll_make_mock_data(nmics, nruns)
print('Done w sim data')

print('First cppyy run')
#prob_u = np.loadtxt('many_u_1051_fails.csv')
# FOR SEED 569 INDEX 303 CREATES CPP ERROR on Win10 laptop
ii = cppyy_sw2002(many_u[0,:], nmics)
print('End cppyy run')

#%%
# #'Warm up' the cppyy function run by calling it once3. 
serial_sta = time.perf_counter_ns()/1e9
all_out = []
for i in tqdm.trange(nruns):
    ##print(f'{i} index')
    try:
        
        bb = cppyy_sw2002(many_u[i,:], nmics)
        all_out.append(bb)
    except:
        print(f'(Working till index {i})')    
        break
cpy_serial = np.array(all_out)
serial_stop = time.perf_counter_ns()/1e9
all_serial = np.array(all_out)
print('Done with serial fun:', serial_stop-serial_sta)
problem_points = np.unique(np.argwhere(cpy_serial==-999)[:,0])

try:
    print(problem_points)
    ind = problem_points[0]
    # run the problem point
    cppyy_sw2002(many_u[ind,:], nmics)
    swo_py(many_u[ind,:], nmics)
    m303 = many_u[problem_points[0],:]
except:
    pass
    #%% Parameter set that throws off Eigen implementations:
tricky = np.array([-4.43944,  -1.60661,   4.00901,
                    5.63564,   1.22122, -4.16416,
                    1.12112,   6.18619,  -2.58258,
                    7.998,   0.17017, -0.950951,
                    -0.45045,  0.660661, -0.710711,
                    -7.10511,   -1.13221,   -5.10312, -0.0929367])
# get closest row
match_dist = np.apply_along_axis(np.linalg.norm, 1, many_u-tricky)
bestfit = np.argmin(match_dist)
print(f'{np.argmin(bestfit)}')
numpy_out_tricky = swo_py(tricky, nmics)
cpy_out_tricky = cppyy_sw2002(tricky, nmics)
#%% Here let's also run the pll version. 

def pll_cppyy_sw2002(many_micntde, many_nmics, num_cores, c):
    block_in = cppyy.gbl.std.vector[cppyy.gbl.std.vector['double']](many_micntde.shape[0])
    block_mics = cppyy.gbl.std.vector[int](many_micntde.shape[0])
    
    for i in range(many_micntde.shape[0]):
        block_in[i] = cppyy.gbl.std.vector['double'](many_micntde[i,:].tolist())
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
def block_numpy_sw2002(many_micntde, many_nmics,c):
    rows, _ = many_micntde.shape
    solutions = np.zeros((rows, 3))
    
    for i in range(rows):
        solutions[i,:] = swo_py(many_micntde[i,:], many_nmics[i])
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
print(f'CPY MAX {np.max(cpy_error)} NUMPY MAX: {np.max(np_error)}')