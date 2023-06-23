# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 17:59:30 2023

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy
import arviz as az
import pymc3 as pm
from tqdm import tqdm
import theano
import theano.tensor as tt
import pandas as pd
import datetime

def Extend_PolicySeries(policy_series):
    policy_series = np.array(policy_series)
    n_steps = policy_series.shape[0]
    n_policies = policy_series.shape[-1]
    policy_series_ = [policy_series]
    policy_series_.extend([np.concatenate([policy_series[i:, :], np.zeros((i, n_policies))], axis=0) \
                           for i in range(1, n_steps)])
    policy_series = np.array(policy_series_)
    # nt * ntau * npolicies
    policy_series.swapaxes(0, 1)
    return policy_series

def Extend_Gdeltat_tau(Gdeltat_tau):
    
    G_tau = tt.zeros_like(Gdeltat_tau)
    n_steps = Gdeltat_tau.shape[1]
    step = 0
    
    def next_step(step, _, Gdeltat_tau):
        G_tau = tt.set_subtensor(Gdeltat_tau[:, 0:step], 0)
        G_tau = tt.roll(G_tau, shift=-step, axis=1)
        return step+1, G_tau
    
    outputs_info = [step, G_tau]
    non_sequences = [Gdeltat_tau]
    outputs, _ = theano.scan(fn=next_step, outputs_info=outputs_info, \
                             non_sequences=non_sequences, n_steps=n_steps)
    
    _, G = outputs
    G = G.dimshuffle(1, 2, 0)
    
    return G

def Extend_Gtau_deltat(Gtau_deltat):
    
    G_tau = tt.zeros_like(Gtau_deltat)
    n_steps = Gtau_deltat.shape[0]
    step = 0
    
    def next_step(step, _, Gtau_deltat):
        G_tau = tt.set_subtensor(Gtau_deltat[0:step, :], 0)
        G_tau = tt.roll(G_tau, shift=-step, axis=0)
        return step+1, G_tau
    
    outputs_info = [step, G_tau]
    non_sequences = [Gtau_deltat]
    outputs, _ = theano.scan(fn=next_step, outputs_info=outputs_info, \
                             non_sequences=non_sequences, n_steps=n_steps)
    
    _, G = outputs
    G = G.dimshuffle(1, 2, 0)
    
    return G

def Triangle_Arrange(X):
    '''
    input: N x N x ... Array
               * * *      * 0 0
    function:  * * *  to  * * 0 
               * * *      * * *
    '''
    
    X = tt.swapaxes(X, 0, 1)
    x = tt.zeros_like(X[0])
    n_steps = X.shape[0]
    index = 0
    
    def next_val(index, _, X):
        x = tt.roll(X[index], shift=index, axis=0)
        x = tt.set_subtensor(x[0:index], 0)
        return index+1, x
    
    outputs_info = [index, x]
    non_sequences = [X]
    outputs, _ = theano.scan(fn=next_val, outputs_info=outputs_info, \
                             non_sequences=non_sequences, n_steps=n_steps)
    
    _, X = outputs
    X = tt.swapaxes(X, 0, 1)
    
    return X

def Layer_Alpha0(R0, Gt, St, it_history, N, policy_series, policy_weights):
    
    policy_series = Extend_PolicySeries(policy_series)
    n_steps = policy_series.shape[0]

    it = tt.zeros_like(tt.constant(np.zeros((n_steps,), dtype=np.float64)))
    It = tt.zeros_like(tt.constant(np.zeros((n_steps,), dtype=np.float64)))
    Itc = tt.zeros_like(tt.constant(np.zeros((n_steps,), dtype=np.float64)))
    
    Rt = tt.zeros_like(tt.constant(np.zeros((n_steps,), dtype=np.float64)))
    
    # tau x t
    it_history = it_history.dimshuffle("x", 0)
    it_history = tt.repeat(it_history, n_steps, axis=0)
    
    # tau
    St = St.dimshuffle("x")
    St = tt.repeat(St, n_steps, axis=0)
    
    def next_day(policy_series, St, it_history, _1, _2, Itc, _3, policy_weights, R0, Gt, N):
        # tau
        Rt = R0 - tt.sum(policy_series * policy_weights, axis=-1)
        # tau x t
        Gt_extend = Gt.dimshuffle("x", 0)
        Gt_extend = tt.repeat(Gt_extend, n_steps, axis=0)
        # tau
        It = tt.sum(Gt_extend * it_history, axis=-1)
        # tau
        Itc = Itc + It
        # tau
        it = Rt / N * St * It
        # update 
        it_history = tt.roll(it_history, shift=1, axis=-1)
        it_history = tt.set_subtensor(it_history[:, 0], it)
        St = St - it
        # for stable
        St = tt.clip(St, 0, N)
        return St, it_history, it, It, Itc, Rt
    
    sequences = [policy_series]
    outputs_info = [St, it_history, it, It, Itc, Rt]
    non_sequences = [policy_weights, R0, Gt, N]
    
    outputs, _ = theano.scan(fn=next_day, sequences=sequences, outputs_info=outputs_info, \
                             non_sequences=non_sequences)
    
    return outputs

def Layer_Alphai(k0, Gtau_deltat, St, it_history, N, policy_series, policy_weights):
    
    # nt * ntau * npolicies
    policy_series = Extend_PolicySeries(policy_series)
    n_steps = policy_series.shape[0]

    it = tt.zeros_like(tt.constant(np.zeros((n_steps,), dtype=np.float64)))
    It = tt.zeros_like(tt.constant(np.zeros((n_steps,), dtype=np.float64)))
    Itc = tt.zeros_like(tt.constant(np.zeros((n_steps,), dtype=np.float64)))
    
    kt = tt.zeros_like(tt.constant(np.zeros((n_steps,), dtype=np.float64)))
    
    # t x tau
    it_history = it_history.dimshuffle(0, "x")
    it_history = tt.repeat(it_history, n_steps, axis=1)
    
    # tau
    St = St.dimshuffle("x")
    St = tt.repeat(St, n_steps, axis=0)
    
    # tau i-1 x deltat x tau i
    Gtau_deltat = Extend_Gtau_deltat(Gtau_deltat)
    Gtau_deltat = Triangle_Arrange(Gtau_deltat)
    
    Gtau_deltat = tt.as_tensor_variable(Gtau_deltat)
    policy_series = tt.as_tensor_variable(policy_series)
    
    def next_day(policy_series, Gtau_deltat, St, it_history, _1, _2, Itc, _3, policy_weights, k0, N):
        # tau
        kt = k0 - tt.sum(policy_series * policy_weights, axis=-1)
        It = tt.sum(Gtau_deltat * it_history, axis=0)
        Itc = Itc + It
        it = kt / N * St * It
        # update 
        it_history = tt.roll(it_history, shift=1, axis=0)
        it_history = tt.set_subtensor(it_history[0, :], it)
        St = St - it
        # for stable
        St = tt.clip(St, 0, N)
        return St, it_history, it, It, Itc, kt
    
    sequences = [policy_series, Gtau_deltat]
    outputs_info = [St, it_history, it, It, Itc, kt]
    non_sequences = [policy_weights, k0, N]
    
    outputs, _ = theano.scan(fn=next_day, sequences=sequences, outputs_info=outputs_info, \
                             non_sequences=non_sequences, n_steps=n_steps)
    
    return outputs
    
def Layer_Last(k0, Gtau_deltat, St, it_history, N, policy_series, policy_weights):
    
    policy_series = np.array(policy_series)
    n_steps = policy_series.shape[0]

    it = tt.zeros_like(St)
    It = tt.zeros_like(St)
    Itc = tt.zeros_like(St)
    
    kt = tt.zeros_like(St)
    
    Gtau_deltat = Triangle_Arrange(Gtau_deltat)
    
    Gtau_deltat = tt.as_tensor_variable(Gtau_deltat)
    policy_series = tt.as_tensor_variable(policy_series)
    
    def next_day(policy_series, Gtau_deltat, St, it_history, _1, _2, Itc, _3, policy_weights, k0, N):
        kt = k0 - tt.sum(policy_series * policy_weights)
        It = tt.sum(Gtau_deltat * it_history)
        Itc = Itc + It
        it = kt / N * St * It
        # update 
        it_history = tt.roll(it_history, shift=1, axis=0)
        it_history = tt.set_subtensor(it_history[0], it)
        St = St - it
        # for stable
        St = tt.clip(St, 0, N)
        return St, it_history, it, It, Itc, kt
    
    sequences = [policy_series, Gtau_deltat]
    outputs_info = [St, it_history, it, It, Itc, kt]
    non_sequences = [policy_weights, k0, N]
    
    outputs, _ = theano.scan(fn=next_day, sequences=sequences, outputs_info=outputs_info, \
                             non_sequences=non_sequences, n_steps=n_steps)
    
    return outputs

def Merge_Layer_Alphai(Xim1_deltat_tau, Xi_deltat_tau):
    # deltat x tau i-1 x tau i
    Xim1_deltat_tau = Extend_Gdeltat_tau(Xim1_deltat_tau)
    Xim1_deltat_tau = Triangle_Arrange(Xim1_deltat_tau)
    Xi_deltat_tau = Xi_deltat_tau.dimshuffle("x", 0)
    Xi_deltat_tau = tt.repeat(Xi_deltat_tau, Xim1_deltat_tau.shape[0], axis=0)
    # deltat x tau i
    return tt.sum(Xim1_deltat_tau*Xi_deltat_tau, axis=1)

def Merge_Layer_Last(Xim1_deltat_tau, Xi_deltat_tau):
    # t x tau
    Xim1_deltat_tau = Triangle_Arrange(Xim1_deltat_tau)
    Xi_deltat_tau = Xi_deltat_tau.dimshuffle("x", 0)
    Xi_deltat_tau = tt.repeat(Xi_deltat_tau, Xim1_deltat_tau.shape[0], axis=0)
    return tt.sum(Xim1_deltat_tau*Xi_deltat_tau, axis=1)

def Convolution(seq, kernel):
    
    n_steps = seq.shape[0]
    padding = tt.zeros_like(kernel[0:-1])
    seq = tt.concatenate([padding, seq], axis=0)
    y = tt.zeros_like(seq[0])
    
    def next_step(seq, _, kernel):
        y = tt.sum(seq[0:kernel.shape[0]] * kernel[::-1])
        seq = tt.roll(seq, shift=-1, axis=0)
        seq = tt.set_subtensor(seq[-1], 0)
        return seq, y
    
    outputs_info = [seq, y]
    non_sequences = [kernel]
        
    outputs, _ = theano.scan(fn=next_step, outputs_info=outputs_info, \
                             non_sequences=non_sequences, n_steps=n_steps)
    _, y = outputs
    return y