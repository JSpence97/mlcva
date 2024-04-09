import samplers.cva_params as cva_params
from scipy.stats import norm
import numpy as np
##### CVA Problem #####
# Useful constants
C_rs = (cva_params.interest + cva_params.sigma**2/2)
C_rms = (cva_params.interest - cva_params.sigma**2/2)

# Useful Functions
def random_choice(M, rng, zeta):  # Returns M random levels for unbiased MLMC
    return np.ceil(-1 - np.log2(1 - rng.uniform(size = M)) / zeta).astype(int)

d1 = lambda t, S, K: (np.log(S/K) + C_rs*(cva_params.T-t))/(cva_params.sigma*np.sqrt(cva_params.T-t))
d2 = lambda t, S, K: (np.log(S/K) + C_rms*(cva_params.T-t))/(cva_params.sigma*np.sqrt(cva_params.T-t))
pi_0 = lambda S:  np.maximum(S - cva_params.K_0, 0)
pi_1 = lambda S:  np.maximum(S - cva_params.K_1, 0)
V_call = lambda t, S, K: norm.cdf(d1(t,S,K))*S - norm.cdf(d2(t,S,K))*K*np.exp(-cva_params.interest*(cva_params.T-t))
dS0_call = lambda t, S, K: S*norm.cdf(d1(t,S,K))/cva_params.S0 \
    + S*norm.pdf(d1(t,S,K))/cva_params.S0/(cva_params.sigma*np.sqrt(cva_params.T-t)) \
    - K*np.exp(-cva_params.interest*(cva_params.T-t))*norm.pdf(d2(t, S, K))/(cva_params.S0*cva_params.sigma*np.sqrt(cva_params.T-t))

# Compute weights of each option
weight_0 = cva_params.val_0/(V_call(0, cva_params.S0, cva_params.K_0) \
    - V_call(0, cva_params.S0, cva_params.K_1)*dS0_call(0, cva_params.S0, cva_params.K_0)/dS0_call(0, cva_params.S0, cva_params.K_1)  )
weight_1 = -weight_0*dS0_call(0, cva_params.S0, cva_params.K_0)/dS0_call(0, cva_params.S0, cva_params.K_1)  # Weight for option pi_1

# Evaluation functions
drift_coeff = lambda t, S: cva_params.interest*S  # Risk-Neutral drift coefficient
diffusion = lambda t, S: cva_params.sigma*S  # Volatility coefficient
der_diffusion = lambda t, S: cva_params.sigma  # Derivative of diffusion w.r.t S

stock_val_risky = lambda  St, t0, t1, noise: St*np.exp((cva_params.drift - cva_params.sigma**2/2)*(t1-t0) \
    + cva_params.sigma*np.sqrt(t1-t0)*noise)
stock_val = lambda  St, t0, t1, noise: St*np.exp((cva_params.interest - cva_params.sigma**2/2)*(t1-t0) \
    + cva_params.sigma*np.sqrt(t1-t0)*noise)
credit_evol = lambda  M, rng: cva_params.c0*np.exp(-cva_params.credit_vol**2/2 * cva_params.risk_horizon \
    + cva_params.credit_vol*np.sqrt(cva_params.risk_horizon)*rng.standard_normal(size = M))

payoff = lambda  St: weight_0*pi_0(St) + weight_1*pi_1(St)

d_payoff = lambda St: St/cva_params.S0 * (weight_0*(St > cva_params.K_0) + weight_1*(St > cva_params.K_1))
EAD = lambda  St, t: np.maximum(weight_0*V_call(t, St, cva_params.K_0) \
    + weight_1*V_call(t, St, cva_params.K_1), 0)
Delta_EAD = lambda  St, t: (weight_0*dS0_call(t, St, cva_params.K_0) \
    + weight_1*dS0_call(t, St, cva_params.K_1))*(EAD(St, t) > 0)
B_t = lambda  t: np.exp(cva_params.interest*t)
N_umlmc = lambda ell: int(np.ceil(cva_params.N0_umlmc*2**ell))

g = lambda x, t: x/cva_params.c0 * np.exp(-(x-cva_params.c0)*t/cva_params.LGD)  # Default control factor

U = lambda S, tau: cva_params.LGD/B_t(tau) * EAD(S, tau)  # CVA value U, conditioned on default



Delta = lambda S, tau : cva_params.LGD/B_t(tau) * Delta_EAD(S, tau)  # Delta CV

sample_default = lambda shape, rng, scale = cva_params.LGD/cva_params.c0, h = 0: \
    h + rng.exponential(scale = scale, size = shape)

rate = cva_params.c0/cva_params.LGD  # Expoential default rate (used below)
prob_ht = np.exp(-rate*cva_params.risk_horizon) - np.exp(-rate*cva_params.T)   # Probability of default in (h,T)
prob_T = 1 - np.exp(-rate*cva_params.T)  # Probability of default in (0, T)
prob_h = 1 - np.exp(-rate*cva_params.risk_horizon)
sample_default_cond_hT = lambda shape, rng: -np.log(np.exp(-rate*cva_params.risk_horizon) \
    + rng.uniform(size = shape)*(np.exp(-rate*cva_params.T) - np.exp(-rate*cva_params.risk_horizon)))/rate
sample_default_cond_h = lambda shape, rng: -np.log(1 + rng.uniform(size = shape)*(np.exp(-rate*cva_params.risk_horizon) - 1))/rate



def default_cv_mc_sampler(M, rng):  # Sampler for monte carlo estimate of default control variate (use change of measure to ensure default)
    tau = sample_default_cond_hT(M, rng)
    return (1/cva_params.c0 - tau/cva_params.LGD)*U(stock_val(cva_params.S0, 0, tau, rng.standard_normal(size = tau.size)), tau)*prob_ht

def before_h_mc_sampler(M, rng):  # Sampler for monte carlo estimate of default control variate (use change of measure to ensure default)
    tau = sample_default_cond_h(M, rng)
    return U(stock_val(cva_params.S0, 0, tau, rng.standard_normal(size = tau.size)), tau)*prob_h

def delta_cv_mc_sampler(M, rng):  # Sampler for monte carlo estimate of default control variate (change of measure to ensure default)
    tau = sample_default_cond_hT(M, rng)
    return Delta(stock_val(cva_params.S0, 0, tau, rng.standard_normal(size=M)), tau)*prob_ht

def Monte_Carlo(sampler, tol, rng, min_samples = 10**3, max_block = int((10**7)/4), p= 0.01):  # Standard monte carlo sampler for control varates (offline work)
    sum0 = 0  # First moments
    sum1 = 0  # Second moments
    count = 0
    Cp = norm.ppf(1-p/2)
    M = min_samples
    while count < M:
        # Sample required no. samples
        new_samples = min(max_block, M - count)
        while new_samples > 0:
            term = sampler(new_samples, rng)
            sum0 += np.sum(term)
            sum1 += np.sum(term**2)
            count += new_samples
            new_samples = min(max_block, M - count)
        var = sum1/count  - (sum0/count)**2
        M = int(var/(tol**2) * Cp**2)+1
        # print('E, M, var: ', sum0/count, M, var)
    return sum0/count

# Lambda term when loss occurs between h and t, with all control variates and exact sampling
Lambda_cv = lambda tau, S_risky, credit_risky, S_p, S_m, S, rng:\
    (g(credit_risky, tau)*np.exp(credit_risky*cva_params.risk_horizon/cva_params.LGD)*U(S, tau) \
        - 0.5*(1 + (credit_risky - cva_params.c0)*(1/cva_params.c0 - tau/cva_params.LGD))*(U(S_p, tau) + U(S_m, tau)) \
        - 0.5*(S_risky - cva_params.S0)*(Delta(S_p, tau) + Delta(S_m, tau)), 0)

### Functions for unbiased sampling of Lambda ###
def sums_payoff(tau, S_p, S_m, S, N, rng):  # Sample the sum of N payoffs given stock vals at default, exact sim of stock
    noise = rng.standard_normal(size = (N, tau.size))
    return np.sum(payoff(stock_val(S_p, tau, cva_params.T, noise)), axis = 0),\
           np.sum(payoff(stock_val(S_m, tau, cva_params.T, noise)), axis = 0),\
           np.sum(payoff(stock_val(S, tau, cva_params.T, noise)), axis = 0)
## Functions for unbiased sampling of Pi ##
def payoff_unbiased(tau, S_p, S_m, S, ell, rng):
    num_steps = int(np.ceil(cva_params.steps0_umlmc*2**ell))  # Number of steps to use at level ell
    dts = (cva_params.T - tau)/num_steps  # dt parameter for level ell (depends on tau)
    root_dts = np.sqrt(dts)
    prob_ell = 2**(-cva_params.zeta_payoff*ell)*(1-2**(-cva_params.zeta_payoff))
    if ell > 0:  # Declare coarse paths if required
        S_p_c = S_p
        S_m_c = S_m
        S_c = S
    for n in range(num_steps):  # Loop over Milstein steps
        t = tau + n*dts
        dw = root_dts*rng.standard_normal(size = (1,tau.size))
        S_p += drift_coeff(t, S_p)*dts + diffusion(t, S_p)*dw\
            + 0.5*der_diffusion(t, S_p)*diffusion(t, S_p)*(dw**2 - dts)
        S_m += drift_coeff(t, S_m)*dts + diffusion(t, S_m)*dw\
            + 0.5*der_diffusion(t, S_m)*diffusion(t, S_m)*(dw**2 - dts)
        S += drift_coeff(t, S)*dts + diffusion(t, S)*dw\
            + 0.5*der_diffusion(t, S)*diffusion(t, S)*(dw**2 - dts)
        if ell > 0:  # Coarse step
            if n%2 == 1:
                S_p_c += drift_coeff(t, S_p_c)*2*dts + diffusion(t, S_p_c)*(dw+dw_m1)\
                    + 0.5*der_diffusion(t,S_p_c)*diffusion(t,S_p_c)*((dw+dw_m1)**2-2*dts)
                S_m_c += drift_coeff(t, S_m_c)*2*dts + diffusion(t, S_m_c)*(dw+dw_m1)\
                    + 0.5*der_diffusion(t,S_m_c)*diffusion(t,S_m_c)*((dw+dw_m1)**2-2*dts)
                S_c += drift_coeff(t, S_c)*2*dts + diffusion(t, S_c)*(dw+dw_m1)\
                    + 0.5*der_diffusion(t, S_c)*diffusion(t,S_c)*((dw+dw_m1)**2-2*dts)
            dw_m1 = dw  # Store previous noise
    # Compute final payoffs
    pi_p = payoff(S_p)
    pi_m = payoff(S_m)
    pi = payoff(S)
    d_pi_p = d_payoff(S_p)
    d_pi_m = d_payoff(S_m)
    if ell > 0:
        pi_p_c = payoff(S_p_c)
        pi_m_c = payoff(S_m_c)
        pi_c = payoff(S_c)
        d_pi_p_c = d_payoff(S_p_c)
        d_pi_m_c = d_payoff(S_m_c)
    else:
        pi_p_c = np.zeros(S_p.shape)
        pi_m_c = np.zeros(S_p.shape)
        pi_c = np.zeros(S_p.shape)
        d_pi_p_c = np.zeros(S_p.shape)
        d_pi_m_c = np.zeros(S_p.shape)


    if S_p.shape[0] > 1:
        return (pi_p[0,:] - pi_p_c[0,:])/prob_ell, (pi_m[0,:] - pi_m_c[0,:])/prob_ell, (pi[0,:] - pi_c[0,:])/prob_ell,\
            (pi_p[1,:] - pi_p_c[1,:])/prob_ell, (pi_m[1,:] - pi_m_c[1,:])/prob_ell, (pi[1,:] - pi_c[1,:])/prob_ell
    else:  # At level 0 of exposure, also return derivative
        return (pi_p[0,:] - pi_p_c[0,:])/prob_ell, (pi_m[0,:] - pi_m_c[0,:])/prob_ell, (pi[0,:] - pi_c[0,:])/prob_ell,\
            (d_pi_p[0,:] - d_pi_p_c[0,:])/prob_ell, (d_pi_m[0,:] - d_pi_m_c[0,:])/prob_ell


def payoff_recursion(tau, S_p, S_m, S, levels, levels_left, ell, rng):
    done = levels_left == ell
    indices = np.where(levels == ell)[1]
    cost = 2*np.sum(done)*3*cva_params.steps0_umlmc*2**ell
    if S_p.shape[0] > 1:
        pi_p = np.zeros(tau.shape)
        pi_m = np.zeros(tau.shape)
        pi = np.zeros(tau.shape)
        pi_p_c = np.zeros(tau.shape)
        pi_m_c = np.zeros(tau.shape)
        pi_c = np.zeros(tau.shape)
        pi_p[done], pi_m[done], pi[done], pi_p_c[done], pi_m_c[done], pi_c[done] =  \
            payoff_unbiased(tau[done], S_p[:,indices], S_m[:,indices], S[:,indices], ell, rng)
        if np.sum(done==False) > 0:
            pi_p[done==False], pi_m[done==False], pi[done==False], pi_p_c[done==False], pi_m_c[done==False], pi_c[done==False], cost_t = \
                payoff_recursion(tau[done==False], S_p, S_m, S, levels, levels_left[done==False], ell+1, rng)
            cost += cost_t
        return pi_p, pi_m, pi, pi_p_c, pi_m_c, pi_c, cost
    else:
        pi_p = np.zeros(tau.shape)
        pi_m = np.zeros(tau.shape)
        pi = np.zeros(tau.shape)
        d_pi_p = np.zeros(tau.shape)
        d_pi_m = np.zeros(tau.shape)
        pi_p[done], pi_m[done], pi[done], d_pi_p[done], d_pi_m[done] =  \
            payoff_unbiased(tau[done], S_p[:,indices], S_m[:,indices], S[:,indices], ell, rng)

        if np.sum(done==False) > 0:
            pi_p[done==False], pi_m[done==False], pi[done==False], d_pi_p[done==False], d_pi_m[done==False], cost_t = \
                payoff_recursion(tau[done==False], S_p, S_m, S, levels, levels_left[done==False], ell+1, rng)
            cost += cost_t
        return pi_p, pi_m, pi, d_pi_p, d_pi_m, cost

def sums_payoff_unbiased(tau, S_p, S_m, S, N, rng):
    umlmc_levels = random_choice((N,tau.size), rng, cva_params.zeta_payoff)
    num_sums = int(3*S_p.shape[0])
    if S_p.shape[0] == 2:  # If we have fine and coarse SDE paths
        num_sums = 6
    else:  # If we have only the fine SDE path (level 0 => requires derivatives of payoff for pi_p and pi_m)
        num_sums = 5
    pis = payoff_recursion(np.repeat(tau.reshape((1, tau.size)), N, axis = 0), S_p, S_m, S, umlmc_levels, umlmc_levels, 0, rng)
    return [np.sum(pis[i], axis = 0) for i in range(num_sums)], pis[-1] + umlmc_levels.size
## --------------------------------------------------- ##

def sample_value(tau, S_p, S_m, S, ell, rng):  # Used to approximate value processes using MC for unbiased estimation of Lambda
    if ell == 0:  # Use 1 block of size 2**(ell) when ell = 0 and two blocks of size 2**(ell-1) otherwise
        blocks = 1
        block_samples = N_umlmc(ell)
    else:
        blocks = 2
        block_samples = N_umlmc(ell-1)
    cost = 0
    val_p = np.zeros((S_p.shape[0], blocks, tau.size))
    val_m = np.zeros((S_p.shape[0], blocks, tau.size))
    val = np.zeros((S_p.shape[0], blocks, tau.size))
    d_val_p = np.zeros((S_p.shape[0], blocks, tau.size))
    d_val_m = np.zeros((S_p.shape[0], blocks, tau.size))
    num_sums = int(3*S_p.shape[0])
    for b in range(blocks):  # For each block
        count = min(10**7, block_samples)  # Cap number of samples computed at once (catch for extreme values of ell)
        num_done = 0
        while count > 0:
            sums, cost_t = sums_payoff_unbiased(tau, S_p, S_m, S, count, rng)
            cost += cost_t
            for i in range(S_p.shape[0]):
                val_p[i, b,:] += sums[0 + i*3]
                val_m[i, b,:] += sums[1 + i*3]
                val[i, b,:] += sums[2 + i*3]
            if S_p.shape[0] == 1:  # If required, add derivatives
                d_val_p[0, b, :] += sums[3]
                d_val_m[0, b, :] += sums[4]
            num_done += count
            count = min(10**7, block_samples - num_done)
    return val_p/block_samples, val_m/block_samples, val/block_samples, d_val_p/block_samples, d_val_m/block_samples, cost

def DU(val, tau, ell):  # CVA value Delta_U with approximate sampling of exposure
    if ell > 0:
        return cva_params.LGD/B_t(cva_params.T) * (np.maximum(np.mean(val[0,:,:], axis=0), 0)\
            -np.mean([np.maximum(val[1, i, :], 0) for i in range(2)], axis = 0))

    else:
        return cva_params.LGD/B_t(cva_params.T) * (np.maximum(np.mean(val[0,:,:], axis = 0), 0))

def Delta_unbiased(val, d_val, tau):  # Delta control variate for unbiased mlmc
     ## Will have different form for alternative payoff ##
    return cva_params.LGD/B_t(cva_params.T) * d_val[0,0,:]*(val[0, 0,:]>0)



def stocks_mil(tau, S_risky, ell, rng): # Simulate stocks using Milstein at levels ell & (ell-1)
    if ell == 0:  # Record whether to store only 1 stock at fine level or 2 stocks (fine and coarse)
        sims = 1
    else:
        sims = 2
    # Allocate space
    S_p = np.zeros((sims, tau.size))
    S_m = np.zeros((sims, tau.size))
    S = np.zeros((sims, tau.size))
    # Exact sim of risk free stock to tau (could be replaced with independent milstein with v. small, fixed, timestep)
    noise = rng.standard_normal(size = (1, tau.size))
    # From 0 to h
    S_p_t = stock_val(cva_params.S0, 0, cva_params.risk_horizon, noise)  # +ve noise
    S_m_t = stock_val(cva_params.S0, 0, cva_params.risk_horizon, -noise)  # -ve noise
    for i in range(sims):
        S_p[i,:]  = S_p_t
        S_m[i,:] = S_m_t
        S[i,:] = S_risky

    # Milstein scheme
    num_steps = int(np.ceil(cva_params.steps0_umlmc_outer*2**ell))
    dts = (tau - cva_params.risk_horizon)/num_steps
    root_dts = np.sqrt(dts)
    for n in range(num_steps):  # Loop over Milstein steps
        t = cva_params.risk_horizon + n*dts
        dw = root_dts*rng.standard_normal(size = tau.size)
        S_p[0,:] += drift_coeff(t, S_p[0,:])*dts + diffusion(t, S_p[0,:])*dw\
            + 0.5*der_diffusion(t, S_p[0,:])*diffusion(t, S_p[0,:])*(dw**2 - dts)
        S_m[0,:] += drift_coeff(t, S_m[0,:])*dts + diffusion(t, S_m[0,:])*dw\
            + 0.5*der_diffusion(t, S_m[0,:])*diffusion(t, S_m[0,:])*(dw**2 - dts)
        S[0,:] += drift_coeff(t, S[0,:])*dts + diffusion(t, S[0,:])*dw\
            + 0.5*der_diffusion(t, S[0,:])*diffusion(t, S[0,:])*(dw**2 - dts)
        if ell > 0:  # Coarse step
            if n%2 == 1:
                S_p[1,:] += drift_coeff(t, S_p[1,:])*2*dts + diffusion(t, S_p[1,:])*(dw+dw_m1)\
                    + 0.5*der_diffusion(t,S_p[1,:])*diffusion(t,S_p[1,:])*((dw+dw_m1)**2-2*dts)
                S_m[1,:] += drift_coeff(t, S_m[1,:])*2*dts + diffusion(t, S_m[1,:])*(dw+dw_m1)\
                    + 0.5*der_diffusion(t,S_m[1,:])*diffusion(t,S_m[1,:])*((dw+dw_m1)**2-2*dts)
                S[1,:] += drift_coeff(t, S[1,:])*2*dts + diffusion(t, S[1,:])*(dw+dw_m1)\
                    + 0.5*der_diffusion(t, S[1,:])*diffusion(t,S[1,:])*((dw+dw_m1)**2-2*dts)
            dw_m1 = dw  # Store previous noise
    return S_p, S_m, S

def delta_cv_umlmc_sampler(M, rng):
    tau = sample_default_cond_hT(M, rng)
    S_p = stocks_mil(tau, cva_params.S0, 0, rng)[0]
    val_t = sample_value(tau, S_p, np.zeros((1,tau.size)), np.zeros((1,tau.size)), 0, rng)
    return Delta_unbiased(val_t[0], val_t[3], tau)*prob_ht

def umlmc_exposure(tau, S_risky, credit_risky, levels, ell, rng):  # Used recursively for UMLMC estimation
    loss = np.zeros(tau.size)  # Store loss term here
    done = levels == ell   # Samples with level ell in UMLMC
    cost = 0
    # Simulate stocks via Milstein to level ell
    S_p, S_m, S = stocks_mil(tau[done], S_risky[done], ell, rng)
    cost += 2*np.sum(done)*(2 + 3*2**ell)  # Num. Gaussian RV's used
    val_p, val_m, val, d_val_p, d_val_m, cost_t = sample_value(tau[done], S_p, S_m, S, ell, rng)  # Sample mean value processes
    cost += cost_t
    loss[done] = g(credit_risky[done], tau[done])*np.exp(credit_risky[done]*cva_params.risk_horizon/cva_params.LGD)\
        *DU(val, tau[done], ell) - 0.5*(1 + (credit_risky[done] - cva_params.c0)*(1/cva_params.c0 - tau[done]/cva_params.LGD))\
        *(DU(val_p, tau[done],  ell) + DU(val_m, tau[done], ell)) # Compute loss and default CV term

    if ell == 0:  # Only subtract delta CV at level 0
        loss[done] -= 0.5*(S_risky[done] - cva_params.S0)*(Delta_unbiased(val_p, d_val_p, tau[done]) + Delta_unbiased(val_m, d_val_m, tau[done]))

    loss[done] *= 2**(cva_params.zeta*ell)/(1-2**(-cva_params.zeta))

    if np.sum(done==False)>0:
        loss[done == False], cost_t = umlmc_exposure(tau[done==False], S_risky[done==False], credit_risky[done==False],\
            levels[done==False], ell+1, rng)
        cost += cost_t

    return loss, cost

# Lambda term when loss occurs between h and t, using control variates and with approximate simulation of the exposure
def Lambda_cv_umlmc(tau, S_risky, credit_risky, S_p, S_m, S, rng):
    umlmc_levels = random_choice((1,tau.size), rng, cva_params.zeta).flatten()
    cost = umlmc_levels.size  # Record number of uniform RV's used
    out, cost_t =  umlmc_exposure(tau, S_risky, credit_risky, umlmc_levels, 0, rng)
    return out, cost + cost_t

def Lambda_cv_umlmc_ell_test(tau, S_risky, credit_risky, S_p, S_m, S, rng, ell = 0):
    umlmc_levels = np.zeros(tau.size) + ell
    cost = umlmc_levels.size  # Record number of uniform RV's used
    out, cost_t =  umlmc_exposure(tau, S_risky, credit_risky, umlmc_levels, 0, rng)
    out /= 2**(cva_params.zeta*ell)/(1-2**(-cva_params.zeta))
    return out, cost + cost_t
### ---------------------------------------- ###
def stocks_an(tau, S_risky, rng):  # Returns all stock values at default, using analytic formula
    # Compute corelated stocks
    noise = rng.standard_normal(size = tau.size)
    # From 0 to h
    S_plus = stock_val(cva_params.S0, 0, cva_params.risk_horizon, noise)  # +ve noise
    S_minus = stock_val(cva_params.S0, 0, cva_params.risk_horizon, -noise)  # -ve noise
    # From h to tau
    noise = rng.standard_normal(size = tau.size)  # Sample new, shared, noise
    S_plus = stock_val(S_plus, cva_params.risk_horizon, tau, noise)
    S_minus = stock_val(S_minus, cva_params.risk_horizon, tau, noise)
    S = stock_val(S_risky, cva_params.risk_horizon, tau, noise)
    return S_plus, S_minus, S, 2*5*tau.size  # Return No. Gaussian Rv's used in final component

def stocks_ub(tau, S_risky, rng):  # Don't simulate stocks here with unbiased simulation
    return 0, 0, 0, 0

def loss_cv(tau, S_risky, credit_risky, rng, stock_sim = stocks_an, lambda_loss = Lambda_cv): # Sample loss for stocks defaulted in (h<tau<T) using all control variates
    S_plus, S_minus, S, cost = stock_sim(tau, S_risky, rng)
    loss_t, cost_t = lambda_loss(tau, S_risky, credit_risky, S_plus, S_minus, S, rng)
    cost += cost_t
    return loss_t, cost

def loss_before_horizon(tau, rng):  # Sample loss for stocks which default before the risk horizon
    return -U(stock_val(cva_params.S0, 0, tau, rng.standard_normal(size = tau.size)), tau)

def loss_full(N, S_risky, credit_risky, rng, loss_h_to_T = loss_cv):
    cost = 0  # Store cost of sampling (Taken as No. Gaussian Rvs used)
    default_times = sample_default_cond_hT((N, S_risky.size), rng) # Random default times
    cost += default_times.size
    ## Future work - Optimize this code
    S_risky = np.repeat(S_risky.reshape((1, S_risky.size)), N, axis = 0)  # Used to track defaulted stocks
    credit_risky = np.repeat(credit_risky.reshape((1, credit_risky.size)), N, axis = 0)
    ##
    #
    loss = np.zeros((N, S_risky.shape[1]))  # Store loss samples
    # Compute loss when h < default < T:
    indices = (cva_params.risk_horizon <= default_times)
    loss[indices], cost_t = loss_h_to_T(default_times[indices], S_risky[indices], credit_risky[indices], rng)
    cost += cost_t
    return loss*prob_ht, cost

def loss_basic(N, S_risky, credit_risky, rng):
    # Sample non_risky default times
    cost = 0  # Store cost of sampling
    default_times = sample_default((N, S_risky.size), rng)
    cost += default_times.size
    # Sample risky default times  ##(FUTURE WORK - optimize section)
    default_risky = np.zeros((N, S_risky.size))
    for j in range(S_risky.size):
        default_risky[:,j:j+1] = sample_default((N,1), rng, scale = cva_params.LGD/credit_risky[j], h = cva_params.risk_horizon)
    ##
    ## Future work - Optimize this code
    S_risky = np.repeat(S_risky.reshape((1, S_risky.size)), N, axis = 0)  # Used to track defaulted stocks
    credit_risky = np.repeat(credit_risky.reshape((1, credit_risky.size)), N, axis = 0)
    ##
    loss = np.zeros((N, S_risky.shape[1]))
    # Risky component:
    index = default_risky <= cva_params.T
    loss[index] = U(stock_val(S_risky[index], cva_params.risk_horizon, default_risky[index], rng.standard_normal(size = np.sum(index))), \
        default_risky[index])
    cost += 2*np.sum(index)
    # Non-risky component
    index = default_times <= cva_params.T
    loss[index] -= U(stock_val(cva_params.S0, 0, default_times[index], rng.standard_normal(size = np.sum(index))), default_times[index])
    cost += 2*np.sum(index)
    return loss, cost
