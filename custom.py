# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 15:21:18 2023

@author: wmy
"""

import pymc3 as pm
import arviz as az
import numpy as np
import matplotlib.pyplot as plt
from pymc3.step_methods.arraystep import BlockedStep

class UpdateIndependentNoiseTermStep(BlockedStep):

    def __init__(self, epsilon, sigma, model=None):
        pm.modelcontext(model)
        self.vars = [epsilon]
        self.epsilon = epsilon
        self.sigma = sigma
        pass

    def step(self, point: dict): 
        sigma = self.sigma
        if not isinstance(1.0*self.sigma, float):
            sigma = np.exp(point[self.sigma.transformed.name])
            pass
        point[self.epsilon.name] = pm.Normal.dist(mu=0, sigma=sigma).random()
        return point
    
    pass