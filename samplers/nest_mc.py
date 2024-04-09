# This file contains classes containing necessary methods used to sample g to compute expectations of the form
#           E[H(E[X|Y])].

# Standard imports
import numpy as np
from numpy.random import default_rng
from scipy.stats import norm
##########

## Nested Expectation - g = \E{X|Y}; noise = (Y, \{X(Y)\}_i). Approximates g by inner MC average
## g_\ell = N_\ell^{-1}\sum_{1<=n<=N_\ell} X^{(n)}(Y), uses antithetic sampling of the multilevel correction term as in
##  [3] Giles, Haji-Ali 'Multilevel nested simulation for efficient risk estimation', 2018.
class noise_nested:  # Stores noise - samples Y, X, X^2. Note: We need X^2 to compute the sample variance for sigma_\ell
    def __init__(self, Y, X, Xsq):
        self.Y = Y  # Initial outer samples
        self.X = X  # Inner samples (shared between levels)
        self.Xsq = Xsq  # Squared inner samples (shared between levels)
        self.size = Y.shape[1]  # Used to determine no. remaining samples when adaptively sampling

class nested_base:
    def __init__(self, N0 = 1, gamma = 1, beta = 1, sigma = lambda self, noise, ell, eta: 1, reuse_samples = True, loss = 0, k0_smooth = None):
        self.N0 = N0  # Number of samples for the inner MC at level 0
        self.gamma = gamma  # Refinement rate
        self.beta = beta # Variance reduction rate
        self.sigma = sigma  # Used in \delta_\ell to measure sample dependent variability (i.e. conditional sample s.d.)
        self.N = lambda ell: int(self.N0*2.**(self.gamma*ell))  # Returns no. inner samples given ell
        self.reuse = reuse_samples  # If False, uses adaptive nested simulation algorithm [3, Algorithm 1]
        self.loss = loss  # Constant loss threshold
        self.k0 = k0_smooth


    def sample_y(self, M):
        raise Exception("Function 'sample_y' has not been defined.")

    def sample_x(self, Y, ell):
        raise Exception("Function 'sample_x' has not been defined.")

    def init_ell(self, ell, ell0):  # Initialise parameters specific to level ell
        self.ell = ell
        if ell == ell0:
            self.ell_m1 = ell
        else:
            self.ell_m1 = ell - 1

    def sample_noise(self, M, rng):  # Initial samples of X, Y at level ell, store X in blocks of sums of N02^{gamma(l-1)} terms
                                # to save memory
        if self.reuse == True:  # If using the general adaptive sampler
            Y = self.sample_y(M, rng)  # Compute Yvals
            if self.ell > self.ell_m1: # Test whether ell > ell - 1 to determine number of blocks required
                num_divs = 2**self.gamma
            else:
                num_divs = 1
            # Compute X and Xsq as blocks of sums of size 2^{gamma*(ell-1)}
            cost = 0
            X = np.zeros((num_divs, M))
            Xsq = np.zeros((num_divs, M))
            for i in range(num_divs):
                x, cost_t = self.sample_x(Y, self.ell_m1, rng)
                cost += cost_t
                X[i, :] = np.sum(x, axis = 0)
                Xsq[i, :] = np.sum(x**2, axis = 0)

            return noise_nested(Y, X, Xsq), cost

        else:
            return self.sample_y(M, rng), 0  # For [3], only sample Y as X will be resampled later

    def split_noise(self, noise, done): # Splits noise into accepted/rejected samples according to method determined by reuse
        if self.reuse == True:
            noise_acc = noise_nested(noise.Y[:, done == True], noise.X[:, done == True], noise.Xsq[:, done == True])
            noise_rej = noise_nested(noise.Y[:, done == False], noise.X[:, done == False], noise.Xsq[:, done == False])
            return noise_acc, noise_rej
        else:
            return noise[:,done==True], noise[:,done==False]


    def refine_noise(self, noise, ell, eta, rng):
        if self.reuse == True: # For the general adaptive algorthm, refine the inner MC algorithm by adding
                               # (2**(gamma) - 1) times more samples of X
            newRows = 2**(self.gamma*(ell - self.ell_m1 + eta))*(2**self.gamma - 1)  # Number blocks of N_{\ell-1} samples
            XNew = np.zeros((newRows, noise.size))
            XsqNew = np.zeros((newRows, noise.size))
            cost = 0 # Store cost of sampling x

            for i in range(newRows):  # Sum to sample each required block
                x, cost_t = self.sample_x(noise.Y, self.ell_m1, rng)
                cost += cost_t
                XNew[i, :] = np.sum(x, axis = 0)
                XsqNew[i, :] = np.sum(x**2, axis = 0)
            noise.X = np.concatenate([noise.X, XNew], axis = 0)
            noise.Xsq = np.concatenate([noise.Xsq, XsqNew], axis = 0)
            return noise, cost

        else:
            return noise, 0  # Not needed in [3] since we resample all X terms

    def sample_g(self, noise, ell, eta, rng): # Compute antithetic means for g
        if self.reuse == True:  # General adaptive sampling
            num_sum = 2**(self.gamma*(ell - self.ell_m1 + eta))  # No. rows of noise.X to use for each mean
            return np.array([np.sum(noise.X[i*num_sum:(i+1)*num_sum, :], axis = 0)/self.N(ell + eta)\
                for i in range(int(noise.X.shape[0]/num_sum))]) - self.loss, 0

        else:  # As in [3]
            if eta < ell - 1:  # Catch for additional term
                # Computes mean of first and second moments of X from samples at level ell+eta
                x_samples, cost = self.sample_x(noise, ell+eta, rng)
                # Return g_{\ell+\eta} in first row, \sigma_{|ell+\eta} in the second and the level in the third
                # to feed information to adaptive sampler
                return np.array((np.mean(x_samples, axis = 0) - self.loss, np.std(x_samples, axis = 0))), cost
            else:
                return np.array([np.zeros((noise.shape[1])), np.ones((noise.shape[1]))]), 0

    def multilevel_correction(self, g_fine, g_coarse, ellf, ellc, noise, rng):
     # Returns antithetic multilevel correction term and associated sampling costs
        if self.reuse == True:
            if type(g_coarse) == str: # If we dont require coarse samples
                return np.mean(g_fine > 0, axis = 0), 0, 0
            else:
                fine = np.mean(g_fine > 0, axis = 0)
                return fine - np.mean(g_coarse > 0, axis = 0), fine, 0, 0

        else:
            if type(g_coarse) == str: # If we dont require coarse samples
                x_samples, cost = self.sample_x(noise, ellf, rng)
                return np.mean(x_samples, axis = 0) > self.loss, cost, cost
            else:
                x_samples, cost = self.sample_x(noise, max(ellf, ellc), rng)
                if ellf >= ellc:  # If the fine level is the most refined, use the same cost for fine estimator, else scale down
                    costf = cost
                else:
                    costf = self.N(ellf)/self.N(ellc) * cost
                Nmax = int(self.N(max(ellf, ellc)))
                fine = np.mean([np.mean(x_samples[i*self.N(ellf):(i+1)*self.N(ellf),:], axis=0)  > self.loss\
                    for i in range(int(Nmax/self.N(ellf)))], axis = 0)   # Antithetic fine estimator
                coarse = np.mean([np.mean(x_samples[i*self.N(ellc):(i+1)*self.N(ellc),:], axis=0)  > self.loss \
                for i in range(int(Nmax/self.N(ellc)))], axis = 0)  # Antithetic coarse estimator
                # Return antithetic mean and costs
                return fine - coarse, fine, cost, costf

    def delta(self, g, noise, ell, eta):
        if self.reuse == True:  # Delta with variable sigma
            return np.abs(g[0, :])/self.sigma(self, noise, ell, eta)  # Must use only a single realisation of g_\ell here
                                                                      # (hence row 0 of g) otherwise we break the
                                                                      # cancellation required for the teloscopic sum in MLMC.
        else:  # Delta using sigma as the sample standard deviation as in [3]
            return np.abs(g[0,:])/np.maximum(g[1,:], 1e-15)
