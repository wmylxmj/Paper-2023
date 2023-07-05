# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 18:35:21 2023

@author: wmy
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy

def LogNormalParams(Ex, Dx):
    sigma2 = np.log(Dx/(Ex**2)+1)
    mu = np.log(Ex) - sigma2/2
    sigma = sigma2**0.5
    return mu, sigma

def truncate_number(number, precision):
    return "{{:.{}f}}".format(precision).format(number)

def median(arr):
    return np.median(arr)

def CI(arr, hdi_prob=0.95):
    half = 100 * (1 - hdi_prob) / 2
    left, right = (np.percentile(arr, q=half), np.percentile(arr, q=100-half))
    return left, right

def HalfCauchy_PDF(x, beta):
    return 2 / (np.pi * beta * (1 + (x/beta) ** 2))

