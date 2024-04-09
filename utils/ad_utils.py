import numpy as np

# Define \sigma_\ell as the sample variance computed from N_{\ell} samples
# Note - Added catch for small variances (for small number of inner samples there is chance no defaults occur, giving
# zero sample variance, which causes problems for the algorithm)
def sigma_sd(self, noise, ell, eta):
    num_pts = self.N0*2**(self.gamma*(ell+eta))  # Number of points at refined level
    indices = 2**(self.gamma*(ell-self.ell_m1 + eta))  # No. blocks of N_{\ell-1} samples to use
    return np.maximum(np.sqrt(np.maximum(np.sum(noise.Xsq[0:indices, :], axis = 0)/num_pts \
        - (np.sum(noise.X[0:indices, :], axis = 0)/num_pts)**2, 0)), 1e-10)
