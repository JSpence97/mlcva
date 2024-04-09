# Standard imports
import numpy as np
from numpy.random import default_rng
from samplers.nest_mc import nested_base
from utils.cva_utils import *

class cva_sampler(nested_base):
    def __init__(self,
        N0 = 1,
        gamma = 1,
        beta = 1,
        sigma = lambda self, noise, ell, eta: 1,
        reuse_samples = True,
        method = 'full',
        cv_estimate_tol = 1e-3,
        def_ctrl = False,
        delta_ctrl = False,
        before_h = False,
        rng_t = default_rng(),
        loss = cva_params.L_eta,
        k0_smoothing = None
    ):
        self.N0 = N0  # Number of samples for the inner MC at level 0
        self.gamma = gamma  # Refinement rate
        self.beta = beta  # Variance reduction rate
        self.sigma = sigma  # Used in \delta_\ell to measure sample dependent variability (i.e. conditional sample s.d.)
        self.N = lambda ell: int(self.N0*2.**(self.gamma*ell))  # Returns no. inner samples given ell
        self.reuse = reuse_samples  # If False, uses adaptive nested simulation algorithm [3, Algorithm 1]
        self.def_ctrl = def_ctrl
        self.delta_ctrl = delta_ctrl
        self.before_h = before_h
        self.loss = loss
        self.k0 = k0_smoothing

        # Set variables according to method:
        if method == 'full':  # All CV's, exact sampling
            self.loss_h_to_T  = lambda tau, S_risky, credit_risky, rng: loss_cv(tau, S_risky, credit_risky, rng)
            if def_ctrl == False:
                print('Estimating default control variate: ', end = ' ', flush = True)
                self.def_ctrl = Monte_Carlo(default_cv_mc_sampler, cv_estimate_tol, rng_t)
                print(self.def_ctrl)
            if delta_ctrl == False:
                print('Estimating delta control variate: ', end = ' ', flush = True)
                self.delta_ctrl = Monte_Carlo(delta_cv_mc_sampler, cv_estimate_tol, rng_t)
                print(self.delta_ctrl)
            if before_h == False:
                print('Estimating Loss before h: ', end = ' ', flush = True)
                self.before_h = Monte_Carlo(before_h_mc_sampler, cv_estimate_tol, rng_t)
                print(self.before_h)
            self.loss_sampler = lambda N, S_risky, credit_risky, rng: loss_full(N, S_risky, credit_risky, rng, \
                loss_h_to_T = self.loss_h_to_T)

        elif method == 'no_cv': # No CV's, exact sampling
            self.before_h = 0
            self.loss_sampler = lambda N, S_risky, credit_risky, rng: loss_basic(N, S_risky, credit_risky, rng)
            self.def_ctrl = 0
            self.delta_ctrl = 0

        elif method == 'approximate_exposure_payoff': # Use unbiased mlmc to compute exposure
            self.loss_h_to_T= lambda tau, S_risky, credit_risky, rng: loss_cv(tau, S_risky, credit_risky, rng,\
                stock_sim = stocks_ub,
                lambda_loss = lambda tau, S_risky, credit_risky, S_p, S_m, S, rng: \
                Lambda_cv_umlmc(tau, S_risky, credit_risky, S_p, S_m, S, rng))
            if def_ctrl == False:
                print('Estimating default control variate: ', end = ' ', flush = True)
                self.def_ctrl = Monte_Carlo(default_cv_mc_sampler, cv_estimate_tol, rng_t)
                print(self.def_ctrl)
            if delta_ctrl == False:
                print('Estimating Delta control variate: ', end = ' ', flush = True)  # Must estimate delta CV at level 0
                self.delta_ctrl = Monte_Carlo(delta_cv_umlmc_sampler, cv_estimate_tol*10, rng_t)
                print(self.delta_ctrl)
            if before_h == False:
                print('Estimating Loss before h: ', end = ' ', flush = True)
                self.before_h = Monte_Carlo(before_h_mc_sampler, cv_estimate_tol, rng_t)
                print(self.before_h)
            self.loss_sampler = lambda N, S_risky, credit_risky, rng: loss_full(N, S_risky, credit_risky, rng, \
                loss_h_to_T = self.loss_h_to_T)

        else:
            raise(ValueError, 'Unrecognised method')



    def sample_y(self, M, rng):
        return np.array([stock_val_risky(cva_params.S0, 0, cva_params.risk_horizon, rng.standard_normal(size = M)),\
            credit_evol(M, rng)])  # Return risky value of stock and credit spread at risk horizon

    def sample_x(self, Y, ell, rng):  # Compute losses arising to change in cva
        N = self.N(ell)
        loss, cost = self.loss_sampler(N, Y[0,:], Y[1,:], rng)  # loss
        loss += (Y[1,:] - cva_params.c0)*self.def_ctrl + (Y[0,:] - cva_params.S0)*self.delta_ctrl - self.before_h  # Add CV's
        return loss, cost
