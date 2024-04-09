import numpy as np
from scipy.stats import norm
from time import perf_counter
from mlmc.adaptive_sampling import adaptive_sampler, det_sampler
from utils.mlmc_data import amlmc_out
from numpy.random import default_rng

class mlmc:
    """
    Class to hold adaptive mlmc level ell sampler, needs to be supplied adaptive rate r (if r <= 0 then uses
    deterministic sampling instead), and sampler class from samplers.py to define problem samples.
    The supplied rate theta (default 1) scales the maximum refined level in adaptive sampling.
    """
    def __init__(self, r = -1, ell0 = 0, c = 1, sampler = None, theta = 1):
        if sampler == None:
            raise Exception("multilevel correction class must be supplied a problem sampler")
        # Declare multilevel correction sampler according to value of r.
        self.ell0 = ell0
        self.gamma = sampler.gamma  # Cost refinement rate
        self.m = 1
        # Define mlmc correction term sampler based on value of r
        if r > 0:
            self.mlmc_sampler = lambda ell, ell0, M, rng: adaptive_sampler(ell, ell0, M, r, c, sampler, theta, rng)
        else:
            self.mlmc_sampler = lambda ell, ell0, M, rng: det_sampler(ell, ell0, M, sampler, rng)
        self.output = amlmc_out() # Initialise output object


    def evaluate(self, ells, M, rng):
        #Implements level l functional of MLMC for purpose of numerical experiments independent of full MLMC computation
        # Initialise params
        print('%-8s%-4s%-15s%-4s%-15s%-4s%-15s%-4s%-15s%-4s%-15s%-4s%-15s%-4s%-15s'\
            %('level','|', 'mean','|', 'variance','|', 'cost','|','mean_f', '|', 'variance_f','|','time','|','kurtosis'))
        print(139*'-')
        for ell in ells:
            cost = 0
            sums = 0
            sumsf = 0
            Vf = 0  # Variance of fine estimator
            mf = 0  # Mean of fine estimator
            t0 = perf_counter()
            # Perform level l mlmc (scale number of samples taken according to level to save memory costs)
            M_temp = int(min(10**6/(2**(self.gamma*ell)), M))
            count = 0
            while count < M:
                term = self.mlmc_sampler(ell, self.ell0, M_temp, rng)
                sums += term[0]
                sumsf += np.array(term[2:4])
                cost += term[-2]
                count += M_temp
                M_temp = int(min(10**6/(2**(self.gamma*ell)), M - count))
            self.output.update(M, ell, cost, sums, sumsf)
            t1 = perf_counter()-t0
            print('%-8i%-4s%-15e%-4s%-15e%-4s%-15f%-4s%-15e%-4s%-15e%-4s%-15e%-4s%-15e'\
                %(ell,'|', self.output.means[ell], '|', self.output.Vs[ell],\
                '|', self.output.costs[ell]/self.output.Ms[ell],'|', self.output.mfs[ell], '|',self.output.Vfs[ell],'|',t1,'|',self.output.kurtosis[ell]))  # Prints useful data

    def mlmc_ell(self, ell, ell0, M, rng):
    #Implements level l functional of mlmc for the MLMC computation in cmlmc defined below
        # Initial params
        cost = 0
        cost_f = 0
        sums = 0
        fine = 0
        fine_sq = 0
        sum_vals = np.zeros(2)
        # Perform level l mlmc (scale number of samples taken according to level to save memory costs)
        M_t = int(max(min(10**6/(2**(self.gamma*ell)), M), 1))
        count = 0
        while count < M:
            term = self.mlmc_sampler(ell, self.ell0, M_t, rng)
            sums += term[0]
            cost += term[-2]
            cost_f += term[-1]
            count += M_t
            fine += term[2]
            fine_sq += term[3]
            sum_vals += term[1]
            M_t = int(max(min(10**6/(2**(self.gamma*ell)), M - count), 1))
        return sums, sum_vals, fine, fine_sq, cost, cost_f

    def evaluate_timed(self, ells, rng, M0 = 1, t_max = 60, filename = None):  # Same as evaluate_par but scales number of samples by max computation time
        t0 = perf_counter()
        M_next = M0
        M_done = 0
        while perf_counter() - t0 < 0.9*t_max:  # Loop until close to max time
            self.evaluate(ells, M_next,  rng)
            print('\n')
            M_done += M_next
            # Estimate possible number of samples in t_max seconds
            M = t_max*M_done / (perf_counter() - t0)
            M_next = int(np.ceil(0.05*M)) # Scale number of samples in next run so as not to overrun time limit
            print('M_done: ', M_done, '  M_next: ', M_next)
            if filename != None:
                self.save(filename)

    def find_l0(self, M, rng, tol_ell0 = 1,  name = None, verbose = 1): # As in [3]: M. B. Giles, A-L. Haji-Ali 'Multilevel nested simulation for efficient risk estimation', 2018.
        # Numerically computes the optimal starting level of mlmc
        # M <- Number samples to estimate variables
        # tol_ell0  <- Tolerance to accept given level (should be >= 1)
        # Initial parameters
        ell0 = -1
        done = False
        if name != None:
            file = open(name, "w")
            file.write("level,R\n")
            count = 0

        while done == False:  # Loop until we find (approximately) optimal starting level
            ell0 += 1
            if verbose == 1:
                print('l0: ', ell0)
            # Obtain MLMC estimates
            ml0 = self.mlmc_ell(ell0, ell0, M, rng)
            ml1 = self.mlmc_ell(ell0+1, ell0, M, rng)

            # Extract relevant parameters
            V0f = ml0[3]/M - (ml0[2]/M)**2  # Variance at level 0
            W0 = ml0[-1]/M  # Work at level 0
            V1 = ml1[0][1]/M - (ml1[0][0]/M)**2  # Variance at level 1
            W1 = ml1[-2]/M  # Work at level 1
            W1f = ml1[-1]/M  # Work of the fine estimator at level 1
            V1f = ml1[3]/M - (ml1[2]/M)**2  # Variance of the fine estimator at level 1
            if name != None:
                file.write(str(ell0) + ',' + str((np.sqrt(V0f*W0) + np.sqrt(V1*W1))/np.sqrt(V1f*W1f)) + "\n")
                count += (np.sqrt(V0f*W0) + np.sqrt(V1*W1)) <= np.sqrt(V1f*W1f)
                done =  (count > 3)
                if count == 1 and ((np.sqrt(V0f*W0) + np.sqrt(V1*W1)) <= tol_ell0 * np.sqrt(V1f*W1f))==1:
                    ell_out = ell0
            # Check optimality within factor given by tol_ell0
            elif (np.sqrt(V0f*W0) + np.sqrt(V1*W1)) <= tol_ell0*np.sqrt(V1f*W1f):
                done = True
                ell_out = ell0
        if name != None:
            file.close()

        if verbose == 1:
            print('\nOptimal l0: ', ell_out, '\n')
        self.ell0 = ell_out


    def bmlmc(self, tols, rng, beta = False, alpha = False, M0 = 10**3, p = 0.05, k =1, err_split = 0.5, lag = 1, \
         bias = True, t_max = 10**10,  verbose = 1, filename = None):
        """
        Performs MLMC computation based on continuation MLMC approach as in [2]:
            tols <- Error tolerances to compute estimator at
            beta, alpha <- multilevel correction variance and bias reduction rate
            M0 <- Number of initial samples at levels ell0, ell0+1, ell0+2
            kap0, kap1 <- Confidence in estimates of proportionality constants
            p <- Desired probability of observing error greater than tol
            err_split <- Proportion of mean square error to be attributed to the bias term
            paramrange <- Number previous levels to compute proportionality constants from
            t_max <- Maximum runtime
        """
        if beta == False or alpha == False:
            raise ValueError('Must declare value of beta and alpha')
        # Initial params
        L = 2  # Start with 3 levels ell = 0,1,2
        ells = np.arange(L+1)
        cost = 0  # Store total cost
        Cp = norm.ppf(1 - p/2)  # Used to scale number of samples per level to ensure the correct error
        lhat = 1  # Initial level for parameter estimation
        t0 = perf_counter()


        # Create data containers
        Mlbar = np.zeros(3)  # Number of samples per level
        suml = np.zeros((4,3))  # Running sum and sum of squares  of terms per level
        costl = np.zeros(3)  # Cost per level
        dMl = (M0*np.ones(3)).astype(int)  # Remaining samples to compute per level
        Vells = np.zeros(3)  # Variances per level
        Eells = np.zeros(3)  # Means per level
        sum_vals = []  # Sum of values = +/- 1  at each level

        # Output containers
        P_out = []  # mlmc_estimate at each tol
        cost_out = []  # Cost at each tol
        L_out = []  # L at each tol
        tol_out = []  # Tolerances computed within runtime
        TOL_prev = 0

        # Run inital hierarchy and update terms
        for ell in range(L + 1):
            sums = self.mlmc_ell(self.ell0 + ell, self.ell0, dMl[ell], rng)
            Mlbar[ell] += dMl[ell]
            suml[:, ell] += sums[0]
            sum_vals.append(sums[1])
            cost += sums[-2]
            costl[ell] += sums[-2]

        # Compute (biased) estimates of Mean/Variance
        for ell in range(L+1):
            if bias == True:
                if ell > lag:  # If we must use regression to estimate true value
                    # Determine parameters b for Beta prior using regression
                    b_var = 0.9*2*k *(self.m + 1)*(2*self.m + 1)* 2**(beta*lag)/6/self.m / Vells[ell - lag] + 1 - 2*self.m*k
                    b_mean = k * (self.m + 1) * 2**(alpha*lag) / 2 / Eells[ell - lag] + 1 - k*self.m
                else:  # Otherwise, use deterministic parameters
                    b_var = 1
                    b_mean = 1
                Eells[ell] = max(np.sum(np.arange(1, self.m + 1)*(sum_vals[ell][:self.m] - sum_vals[ell][self.m:] + k)),\
                    np.sum(np.arange(1, self.m + 1)*(sum_vals[ell][self.m:] - sum_vals[ell][:self.m] + k))) / self.m / \
                    (Mlbar[ell] + self.m*k + b_mean - 1)
                Vells[ell] = np.sum([(i+1)**2 / self.m**2 * (sum_vals[ell][i] + sum_vals[ell][i + self.m] + 2*k) \
                    / (Mlbar[ell] + 2*self.m*k + b_var - 1) for i in range(self.m)])

            else:
                Eells[ell] = suml[0, ell]/Mlbar[ell]
                Vells[ell] =  suml[1,ell]/Mlbar[ell] - (suml[0,ell]/Mlbar[ell])**2

        # Loop over all error tolerances
        for TOL in tols:
            if (TOL_prev / TOL)**2 * (perf_counter() - t0) > t_max:  # Check if we expect the next computation to exceed maximum runtime
                break
            # Compute optimal # samples per level
            Ml = np.ceil((Cp/err_split/TOL)**2*np.sqrt(Vells/(costl/Mlbar))\
                *np.sum(np.sqrt(Vells*costl/Mlbar)))
            dMl = np.maximum(Ml - Mlbar,0).astype(int)
            if verbose == 1:
                print('\nTOL: ', TOL)
            while np.sum(dMl > 0):  # Loop until convergence criteria met
                # Display useful data
                if verbose == 1:
                    print('Time elapsed: ', perf_counter() - t0)
                for ell in range(L+1):
                    if verbose == 1:
                        print('%-8i%-20i'%(ell,dMl[ell]))
                if verbose == 1:
                    print('\n')
                    print('%-8s%-4s%-20s%-4s%-20s%-4s%-20s%-4s%-20s%-4s%-20s%-4s%-20s'\
                        %('level','|', 'mean','|', 'mean est','|','variance','|', 'variance est','|', 'cost','|','M_ell'))
                    print(152*'-')

                # Run mlmc hierarchy
                for ell in range(L + 1):
                    if dMl[ell] > 0:
                        if perf_counter() - t0 > t_max:  # Break if execution time too long
                            break
                        sums = self.mlmc_ell(self.ell0 + ell, self.ell0, dMl[ell], rng)  # Obtain mlmc terms with dMl samples
                        Mlbar[ell] += dMl[ell]  # Update number of samples computed at level ell
                        suml[:, ell] += sums[0]  # Update sum and sum of squared terms at level ell
                        cost += sums[-2]  # Update total cost and cost at level ell
                        costl[ell] += sums[-2]
                        sum_vals[ell] += sums[1]
                        # print('vals: ', np.linspace(-1, 1, 2*self.m))
                        # print('sum vals: ', sum_vals[ell])
                    if bias == True:
                        if ell > lag:  # If we must use regression to estimate true value
                            # Determine parameters b for Beta prior using regression
                            b_var = 0.9*2*k *(self.m + 1)*(2*self.m + 1)* 2**(beta*lag)/6/self.m / Vells[ell - lag] + 1 - 2*self.m*k
                            b_mean = k * (self.m + 1) * 2**(alpha*lag) / 2 / Eells[ell - lag] + 1 - k*self.m
                        else:  # Otherwise, use deterministic parameters
                            b_var = 1
                            b_mean = 1
                        Eells[ell] = max(np.sum(np.arange(1, self.m + 1)*(sum_vals[ell][:self.m] - sum_vals[ell][self.m:] + k)),\
                            np.sum(np.arange(1, self.m + 1)*(sum_vals[ell][self.m:] - sum_vals[ell][:self.m] + k))) / self.m / \
                            (Mlbar[ell] + self.m*k + b_mean - 1)
                        Vells[ell] = np.sum([(i+1)**2 / self.m**2 * (sum_vals[ell][i] + sum_vals[ell][i + self.m] + 2*k) \
                            / (Mlbar[ell] + 2*self.m*k + b_var - 1) for i in range(self.m)])

                    else:
                        Eells[ell] = suml[0, ell]/Mlbar[ell]
                        Vells[ell] =  suml[1,ell]/Mlbar[ell] - (suml[0,ell]/Mlbar[ell])**2
                    if verbose == 1:
                        print('%-8i%-4s%-20e%-4s%-20e%-4s%-20e%-4s%-20e%-4s%-20f%-4s%-20i'\
                            %(ell,'|', suml[0, ell]/Mlbar[ell],'|',Eells[ell], '|', suml[1,ell]/Mlbar[ell] - (suml[0,ell]/Mlbar[ell])**2,\
                            '|', Vells[ell], '|', costl[ell]/Mlbar[ell],'|', Mlbar[ell]))  # Prints useful data
                if perf_counter() - t0 > t_max:
                    break
                TOL_prev = TOL

                # (Re-)Compute optimal # samples per level
                Cl = costl/Mlbar
                Ml = np.ceil((Cp/err_split/TOL)**2*np.sqrt(Vells/Cl)\
                    *np.sum(np.sqrt(Vells*Cl)))
                dMl = np.maximum(Ml - Mlbar,0).astype(int)

                if np.sum(dMl>0.01*Mlbar) == 0: # Test for convergence if computed near optimal samples
                    ERR = abs(Eells[-1]) / (2**alpha - 1) + Cp*np.sqrt(np.sum(Vells/Mlbar))  # Estimate bias (want E_L < TOL / 2)
                    if verbose == 1:
                        print('ERR: ', ERR)

                    if ERR > TOL:  # If the error is still too large, append a new level
                        L += 1
                        # Append new level to params #####
                        Vells = np.append(Vells, Vells[-1]/2**beta)
                        Eells = np.append(Eells, 0)
                        Mlbar = np.append(Mlbar, 0)
                        suml = np.append(suml, np.array([[0],[0],[0],[0]]), axis = 1)
                        ells = np.append(ells, L)
                        costl = np.append(costl, 0)
                        sum_vals.append(np.zeros(2*self.m))
                        ##########
                        # Compute new number of samples for each level
                        Cl = np.append(Cl, 2**self.gamma*Cl[-1])
                        Ml = np.ceil((Cp/err_split/TOL)**2*np.sqrt(Vells/Cl)\
                            *np.sum(np.sqrt(Vells*Cl)))
                        dMl = np.maximum(Ml - Mlbar,0)
            if perf_counter() - t0 > t_max:
                break
            P_out.append(np.sum(suml[0,:]/Mlbar))
            cost_out.append(cost)
            L_out.append(L)
            tol_out.append(TOL)
            if verbose == 1:
                print('Time elapsed: ', perf_counter() - t0)
                print('Estimate: ', P_out[-1], '\n\n'+ 120*'#', '\n')

            # Store data
            self.output.P = P_out
            self.output.cost_mlmc = cost_out
            self.output.L_mlmc = L_out
            self.output.tol_mlmc = tol_out

            if filename != None:
                self.save_mlmc(filename)

    def save_mlmc(self, title):
        # Writes parameters from cmlmc computation to file 'title'
        numTol = len(self.output.tol_mlmc)
        file = open(title, "w")
        file.write('tol,P,cost,L\n')
        for i in range(numTol):
            file.write(str(self.output.tol_mlmc[i]) + ',' + str(self.output.P[i]) \
            + ',' + str(self.output.cost_mlmc[i]) + ',' + str(self.output.L_mlmc[i]) + '\n')
        file.close()

    def save(self, title):
        # Write output data to file 'title'
        file = open(title, "w")
        file.write("level,M,cost,Vf,V,mf,m,kurtosis\n")
        for ell in self.output.levels:
            file.write(str(self.output.levels[ell]) + ',' + str(self.output.Ms[ell])\
             + ',' + str(self.output.costs[ell]) + ',' + str(self.output.Vfs[ell]) + ',' + str(self.output.Vs[ell]) + ',' + \
             str(self.output.mfs[ell]) + ',' +  str(self.output.means[ell]) + ',' + str(self.output.kurtosis[ell]) +  "\n")
        file.close()
