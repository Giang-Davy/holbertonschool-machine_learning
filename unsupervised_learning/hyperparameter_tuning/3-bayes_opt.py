#!/usr/bin/env python3
"""3-bayes_opt.py"""


import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """class de l'optimisation bayesien"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """initialisation"""
        self.f = f
        self.gp = GP(X_init, Y_init, l=1, sigma_f=1)
        ac_samples_points = np.linspace(bounds[0], bounds[1], ac_samples)
        self.X_s = ac_samples_points.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
