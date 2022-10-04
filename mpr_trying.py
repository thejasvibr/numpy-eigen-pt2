# -*- coding: utf-8 -*-
"""
Huang et al. and Scheuing DATEMM code implementation 
====================================================
Created on Tue Oct  4 07:27:00 2022

@author: theja
"""
import numpy as np 
import time
from joblib import Parallel, delayed
import os
import tqdm
from pydatemm import localisation_mpr2003 as mpr
from comparison import make_mock_data, make_block_mock_data, pll_make_mock_data
from sw2002_vectorbased_pll import get_nmics
import scipy.spatial as spl
np.random.seed(78464)
nmics = 4
nruns = int(1e4)
tde_data, source = aa = pll_make_mock_data(nmics, nruns)
#%% Mellen Pachter 2003
def mpr_wrapper(tde_data):
    nmics = get_nmics(tde_data)
    micxyz = tde_data[:nmics*3].reshape(-1,3)
    d = tde_data[-(nmics-1):]
    output = mpr.mellen_pachter_raquet_2003(micxyz, d)
    return output

def mpr_chunk(tde_data):
    outputs = [mpr_wrapper(tde_data[i,:]) for i in range(tde_data.shape[0])]
    return np.row_stack(outputs)

def pll_mpt(many_tde):
    cores = os.cpu_count()
    data_chunks = np.array_split(many_tde, cores)
    outputs = Parallel(n_jobs=cores)(delayed(mpr_chunk)(chunk)for chunk in data_chunks)
    return np.row_stack(outputs)
    
serial_sta = time.perf_counter_ns()/1e9
serial_out = []
for i in tqdm.trange(nruns):
    serial_out.append(mpr_wrapper(tde_data[i,:]))
serial_out = np.row_stack(serial_out)
serial_sto = time.perf_counter_ns()/1e9
serial_durn = serial_sto-serial_sta
print(f'Serial time : {serial_durn}, per-run: {(serial_durn)/nruns}')

pll_sta = time.perf_counter_ns()/1e9
pll_out = pll_mpt(tde_data)
pll_sto = time.perf_counter_ns()/1e9
pll_durn = pll_sto - pll_sta
print(f'Parallel time : {pll_durn}, per-run: {(pll_durn)/nruns}')