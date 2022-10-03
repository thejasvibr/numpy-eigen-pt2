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
vector = cppyy.gbl.std.vector
print(f'{compile_stop-compile_start} s for compilation')


from sw2002_vectorbased_pll import sw_matrix_optim as swo_py
# make neat wrapper. 
#%%
def cppyy_sw2002(micntde):
    as_Vxd = cppyy.gbl.sw_matrix_optim(cppyy.gbl.std.vector['double'](micntde.tolist()),
                                       )
    return np.array(as_Vxd, dtype=np.float64)

nruns = int(1e6)
np.random.seed(569)
nmics = 25
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
ii = cppyy_sw2002(many_u[0,:])
print('End cppyy run')

#%%
# #'Warm up' the cppyy function run by calling it once3. 
serial_sta = time.perf_counter_ns()/1e9
all_out = []
for i in tqdm.trange(nruns):
    ##print(f'{i} index')
    try:
        
        bb = cppyy_sw2002(many_u[i,:])
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
    cppyy_sw2002(many_u[ind,:])
    swo_py(many_u[ind,:])
    m303 = many_u[problem_points[0],:]
except:
    pass

#%% Here let's also run the pll version. 

def pll_cppyy_sw2002(many_micntde, num_cores, c):
    block_in = cppyy.gbl.std.vector[cppyy.gbl.std.vector['double']](many_micntde.shape[0])
    
    for i in range(many_micntde.shape[0]):
        block_in[i] = cppyy.gbl.std.vector['double'](many_micntde[i,:].tolist())
    block_out = cppyy.gbl.pll_sw_optim(block_in, num_cores, c)
    pred_sources = np.array([each for each in block_out])
    return pred_sources

print('Starting pll cppyy run')
ncores = os.cpu_count()
st = time.perf_counter_ns()
uu = pll_cppyy_sw2002(many_u, ncores, 343.0)
stop = time.perf_counter_ns()
pll_durn = (stop-st)/1e9
print(f'OMP pll version takes: {pll_durn} s')
# print(f'Pll vs Serial speedup : {avg_cpy/(pll_durn/nruns)}')

#%%
def block_numpy_sw2002(many_micntde,c):
    rows, _ = many_micntde.shape
    solutions = np.zeros((rows, 3))
    
    for i in range(rows):
        solutions[i,:] = swo_py(many_micntde[i,:])
    return solutions

def pll_numpy_sw2002(many_micntde, c):
    numcores = os.cpu_count();
    indices = np.array_split(np.arange(many_micntde.shape[0]), numcores)
    blocks_micntde = [many_micntde[block,:] for block in indices]
    outputs = Parallel(n_jobs=numcores)(delayed(block_numpy_sw2002)(tde_block,c) for tde_block in blocks_micntde)
    all_out = np.row_stack(outputs)
    return all_out

print('Starting Numpy pll run')
sta = time.perf_counter_ns()/1e9
np_pll = pll_numpy_sw2002(many_u, 343.0)
sto = time.perf_counter_ns()/1e9
pll_np = sto-sta
print(f'Python Pll total durn: {sto-sta} s')


#%%
print(f' OMP C++ vs Joblib Python: {pll_durn , pll_np}')
print(f'C++ advantage is: {pll_np/pll_durn}')
    
#%% Now check the accuracy of outputs
cpy_error = np.array([np.linalg.norm(sources[each,:]-uu[each,:]) for each in range(nruns)])

np_error = np.array([np.linalg.norm(sources[each,:]-np_pll[each,:]) for each in range(nruns)])
print(f'CPY MAX {np.max(cpy_error)} NUMPY MAX: {np.max(np_error)}')

print(f'Avg time pll cpp:{pll_durn/nruns}, np: {pll_np/nruns}')
