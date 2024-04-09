import numpy as np

class amlmc_out:  # Class to store output data from mlmc
    def __init__(self):
        self.levels = []  # Levels used
        self.costs = []  # Average sampling cost per level
        self.Vfs = []  # Variance of the fine estimators
        self.Vs = []  # Variance of the multilevel correction terms
        self.mfs = []  # Mean of the fine estimators
        self.means = []  # Mean of the multilevel correction terms
        self.sums = []  # Continuous sum for first two moments of multilevel correction terms
        self.sumsf = []  # Continuous sum for first two moments of fine terms
        self.Ms = []  # Number samples used per level
        self.kurtosis = []
        self.started_rf = False  # Records if previously been evaluated for root finding purpose (stores previous computation if so)

    def update(self, M, ell, cost, sums, sumsf):  # Used to update data due to new terms
        if ell in self.levels: # If ell has already been considered, update existing terms
            index = self.levels.index(ell)
        else: # Otherwise append to existing results
            self.levels.append(ell)
            self.costs.append(0)
            self.Vfs.append(0)
            self.Vs.append(0)
            self.mfs.append(0)
            self.means.append(0)
            self.sums.append(np.zeros(4))
            self.sumsf.append(np.zeros(2))
            self.Ms.append(0)
            self.kurtosis.append(0)
            index = -1
        self.sums[index] += sums
        self.sumsf[index] += sumsf
        self.costs[index] += cost
        self.Ms[index] += M
        self.Vfs[index] = self.sumsf[index][1]/self.Ms[index] - (self.sumsf[index][0]/self.Ms[index])**2
        self.Vs[index] = self.sums[index][1]/self.Ms[index] - (self.sums[index][0]/self.Ms[index])**2
        self.mfs[index] = self.sumsf[index][0]/self.Ms[index]
        self.means[index] = self.sums[index][0]/self.Ms[index]
        self.kurtosis[index] = (self.sums[index][3]/self.Ms[index] - 4*self.sums[index][2]*self.means[index]/self.Ms[index]\
            + 6*self.sums[index][1]*self.means[index]**2/self.Ms[index] - 3*self.means[index]**4)/(self.Vs[index]**2)
