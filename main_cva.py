import numpy as np
from numpy.random import default_rng
import os
from time import perf_counter

from samplers.cva_sampler import cva_sampler
from utils.ad_utils import sigma_sd
from mlmc.mlmc import mlmc


def main(config, t_max = 120):  # t_max = maximum runtime
    rng = default_rng(73259327732874)
    folder = os.path.join('data', config.name)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok = True)

    tols = 2.**(-np.arange(0, 12))*0.01 # Tolerances

    t0 = perf_counter()

    sampler = cva_sampler(
        N0 = config.N_00,
        gamma = config.gamma,
        beta = config.beta,
        sigma = sigma_sd,
        reuse_samples = True,  # If true, use adaptive MLMC in [2], else use adaptive algorithm in [1]
        cv_estimate_tol = 1e-5,
        method = 'approximate_exposure_payoff',  # Approximate assets and exposure using nested MLMC
        rng_t = rng
    )
    print('Calculating Multilevel Correction Statistics')
    # Pre compute level ell statistics for required levels for independent results

    mlmc_estimator = mlmc(
        r = config.R,
        c = 3/(config.N_00**0.5), # Confidence constant to match that in [1]
        sampler = sampler
    )

    if config.mode == 'level_stats':
        mlmc_estimator.evaluate_timed(
            np.arange(config.L_test + 1).astype(int),
            rng,
            M0 = 100,
            t_max = t_max + t0 - perf_counter(),  # Use 50% remaining time to compute MLMC statistics
            filename = os.path.join(folder,'level_statistics.csv')
        )

    elif config.mode == 'mlmc':
        # Find optimal starting level
        mlmc_estimator.find_l0(
            4096,
            rng,
            tol_ell0 = 1.05,
            name = os.path.join(folder, 'ell0_computation.csv')
        )

        # Perform MLMC Computation and save data
        mlmc_estimator.bmlmc(
            tols,
            rng,
            beta = config.beta,
            alpha = config.alpha,
            err_split = 0.5,  # 50% total error attributed each to statistical variance and to approximation bias
            p = 0.05,  # 95% confidence level
            t_max = t_max + t0 - perf_counter(),
            filename = os.path.join(folder,'mlmc_statistics.csv'),
            lag = 1  # Lag used to estimate statistics of mlmc terms
        )

if __name__ == '__main__':
    from utils.cva_config import parser
    config = parser.parse_args()

    main(config, t_max = config.t_max)

# [1]: Giles, M and Haji-Ali, A.-L. "Efficient multilevel nested simulation", 2019
# [2]: Haji-Ali, A.-L. and Spence, J. and Teckentrup, A. "Adaptive multilevel Monte Carlo for probabilities", 2022
