# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 18:57:50 2023

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
import arviz as az
import pymc3 as pm
from tqdm import tqdm
import theano
import theano.tensor as tt
import pandas as pd
import datetime

def SIR_Model(beta0, gamma, S_begin, I_begin, N, policy_series, policy_weights):
    
    infected_cases_new = tt.zeros_like(I_begin)
    recovered_cases_new = tt.zeros_like(I_begin)
    betat = tt.zeros_like(I_begin)
    
    def next_day(policy_series, S, I, _1, _2, _3, policy_weights, beta0, gamma, N):
        betat = beta0 - tt.sum(policy_series * policy_weights)
        infected_cases_new = betat / N * I * S
        recovered_cases_new = gamma * I
        S = S - infected_cases_new
        I = I + infected_cases_new - recovered_cases_new
        # for stability
        I = tt.clip(I, 0, N)
        return S, I, infected_cases_new, recovered_cases_new, betat
    
    sequences = [policy_series]
    outputs_info=[S_begin, I_begin, infected_cases_new, recovered_cases_new, betat]
    non_sequences=[policy_weights, beta0, gamma, N]
    
    outputs, _ = theano.scan(fn=next_day, sequences=sequences, outputs_info=outputs_info, \
                             non_sequences=non_sequences)
    
    return outputs

def SEIR_Model(beta0, sigma, gamma, S_begin, E_begin, I_begin, N, policy_series, policy_weights):
    
    infected_new = tt.zeros_like(I_begin)
    exposed_new = tt.zeros_like(I_begin)
    removed_new = tt.zeros_like(I_begin)
    betat = tt.zeros_like(I_begin)
    
    def next_day(policy_series, S, E, I, _1, _2, _3, _4, policy_weights, beta0, sigma, gamma, N):
        betat = beta0 - tt.sum(policy_series * policy_weights)
        exposed_new = betat / N * I * S
        infected_new = sigma * E
        removed_new = gamma * I
        S = S - exposed_new
        E = E + exposed_new - infected_new
        I = I + infected_new - removed_new
        # for stability
        S = tt.clip(S, 0, N)
        E = tt.clip(E, 0, N)
        I = tt.clip(I, 0, N)
        return S, E, I, exposed_new, infected_new, removed_new, betat
    
    sequences = [policy_series]
    outputs_info=[S_begin, E_begin, I_begin, exposed_new, infected_new, removed_new, betat]
    non_sequences=[policy_weights, beta0, sigma, gamma, N]
    
    outputs, _ = theano.scan(fn=next_day, sequences=sequences, outputs_info=outputs_info, \
                             non_sequences=non_sequences)
    
    return outputs

def Regeneration_Model(R0, Gt, St, it_history, N, policy_series, policy_weights):
    
    policy_series = np.array(policy_series)
    n_steps = policy_series.shape[0]
    
    it = tt.zeros_like(St)
    It = tt.zeros_like(St)
    Itc = tt.zeros_like(St)
    
    Rt = tt.zeros_like(St)
    policy_series = tt.as_tensor_variable(policy_series)
    
    def next_day(policy_series, St, it_history, _1, _2, Itc, _3, policy_weights, R0, Gt, N):
        Rt = R0 - tt.sum(policy_series * policy_weights)
        It = tt.sum(it_history * Gt)
        Itc = Itc + It
        it = Rt / N * St * It
        it_history = tt.concatenate([tt.reshape(it, newshape=(1,)), it_history[:-1]], axis=0)
        St = St - it
        St = tt.clip(St, 0, N)
        return St, it_history, it, It, Itc, Rt
    
    sequences = [policy_series]
    outputs_info = [St, it_history, it, It, Itc, Rt]
    non_sequences = [policy_weights, R0, Gt, N]
    
    outputs, _ = theano.scan(fn=next_day, sequences=sequences, outputs_info=outputs_info, \
                             non_sequences=non_sequences)
    
    return outputs

