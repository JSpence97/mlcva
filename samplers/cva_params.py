## Constants used in CVA sampler

risk_horizon = 10/365  # Risk horizon H
drift = 0.1  # Geometric Brownian motion drift
sigma = 0.1  # Geometric Brownian motion volatility
interest = 0.01  # Constant interest rate
T = 1  # Maturity
S0 = 1  # Initial asset price
K_0 = 0.4  # First Strike Price
K_1  = 1.1  # Second strike Price
c0 = 500e-4  # Initial credit spread
LGD = 0.6  # Loss given default
L_eta = 5e-4
credit_vol = 0.008/c0
zeta = 9./8. # Unbiased MLMC probability rate
zeta_payoff = 5./4.
N0_umlmc = 1 # Number samples at nested level 0
steps0_umlmc = 1 # Number Milstein steps at nested approximation levels
steps0_umlmc_outer = 1 # Number Milstein steps at outer approximation loop
val_0 = 0.4  # Value of the contract at time 0
