#!/usr/bin/env python3
"""4-bayes_opt.py"""


import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """class de l'optimisation bayesien"""
    def __init__(self, f, X_init, Y_init, bounds, ac_samples,
                 l=1, sigma_f=1, xsi=0.01, minimize=True):
        """initialisation"""
        self.f = f
        self.gp = GP(X_init, Y_init, l=l, sigma_f=sigma_f)
        ac_samples_points = np.linspace(bounds[0], bounds[1], ac_samples)
        self.X_s = ac_samples_points.reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """acquisition"""
        mu, sigma = self.gp.predict(self.X_s)
        ecart = sigma.flatten()
        if self.minimize:
            Z = (min(self.gp.Y) - mu - self.xsi) / ecart
            EI = (min(
                self.gp.Y) - mu - self.xsi) * norm.cdf(Z) + ecart * norm.pdf(Z)
        else:
            Z = (mu - max(self.gp.Y) - self.xsi) / ecart
            EI = (mu - max(
                self.gp.Y) - self.xsi) * norm.cdf(Z) + ecart * norm.pdf(Z)
        X_next = self.X_s[np.argmax(EI)].reshape(1,)

        return X_next, EI
