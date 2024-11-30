import numpy as np
from SamplerUtils import *
import matplotlib.pyplot as plt
import argparse


def SampleNotSymmetrized(N_target,seed,input_name,Nper):
    ### only a small fraction of the LHC samples will be in interesting range
    ### of 0.05-15.0 kbT

    yield_rate = 0.1
    N_total = int(N_target/yield_rate)

    unred = get_LHC_adjusted(seed, N_total)
    N_total = len(unred)
    n_loop = int(N_total//Nper) + 3
    en_unred = np.zeros(N_total,dtype=np.float32)


    unred = unred.astype(np.float32)
    for i in range(n_loop):
        # print('%d/%d'%(i,n_loop))
        if((i+1)*Nper < N_total):
            en_unred[i*Nper:(i+1)*Nper] = calc_en_unred(unred[i*Nper:(i+1)*Nper])
        else:
            en_unred[i*Nper:] = calc_en_unred(unred[i*Nper:])
            break

    en_unred[en_unred>15.0] = 15.0 # get rid of infs and NaNs

    savename_x = './datadir/x_NotSymmetrized_' + input_name + '.npy'
    savename_y = './datadir/y_' + input_name + '.npy'

    np.save(savename_x,unred)
    np.save(savename_y,en_unred)

def ConvertNotSymmetrizedToRaw(input_name):
    fname = './datadir/x_NotSymmetrized_' + input_name + '.npy'
    not_symmetrized = np.load(fname)
    raw = unred_to_raw(not_symmetrized)
    raw = raw.astype(np.float32)

    savename = './datadir/x_Raw_' + input_name + '.npy'
    np.save(savename,raw)

def ConvertRawToSymmetrized(input_name,N_per):
    fname = './datadir/x_Raw_' + input_name + '.npy'
    raw = np.load(fname)

    N_total = len(raw)
    sym = np.zeros((N_total,6),dtype=np.float32)
    n_loop = int(N_total//N_per) + 3

    for i in range(n_loop):
        # print('%d/%d'%(i,n_loop))
        if((i+1)*N_per < N_total):
            sym[i*N_per:(i+1)*N_per] = raw_to_red(raw[i*N_per:(i+1)*N_per])
        else:
            sym[i*N_per:] = raw_to_red(raw[i*N_per:])
            break


    sym = sym.astype(np.float32)
    savename = './datadir/x_Symmetrized_' + input_name + '.npy'
    np.save(savename,sym)


def ConvertNotSymmetrizedTo12pts(input_name):
    fname = './datadir/x_NotSymmetrized_' + input_name + '.npy'
    not_symmetrized = np.load(fname)
    pts12 = unred_to_12pts(not_symmetrized)
    pts12 = pts12.astype(np.float32)

    savename = './datadir/x_12pts_' + input_name + '.npy'
    np.save(savename,pts12)


def dummy():
    pass
