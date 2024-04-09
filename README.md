# Efficient Risk Estimation for the Credit Valuation Adjustment

This repository contains code used to approximates the probability of large loss owed to fluctuations in the credit valuation adjustment (CVA), as discussed in the paper:

[1] "Efficient Risk Estimation for the Credit Valuation Adjustment", Giles M., Haji-Ali, A.-L. and Spence, J., 2024, available at https://arxiv.org/abs/2301.05886.

# Files

#### MLMC

`adaptive_sampling.py` contains functions for adaptive refinements within MLMC
used in [1]

`mlmc.py` contains a base mlmc class which can compute MLMC estimates and statistics of
multilevel correction terms.

#### Samplers

`cva_params.py` assigns values to various constant parameters within the credit
valuation adjustment model problem considered in [1]

`nest_mc.py` contains a base class for sampling nested (multilevel) Monte Carlo estimates

`cva_sampler.py` contains a class which generates samples of multilevel differences for
the CVA model problem.

#### Utils
`cva_utils.py` contains functions used for sampling market and default parameters within the
nested MLMC framework for the CVA problem.

`ad_utils.py` contains a class which samples the parameter sigma_l in [1], defined as the standard deviation
of nested (unbiased) Monte Carlo estimates.

`mlmc_data.py` contains a class used to store statistics of MLMC correction terms.


# Usage
The experiments in [1] are produced using the file `main_cva.py`, which can be
supplied the following commands:

--N_00: The number of nested MC samples at level 0
--gamma: The geometric rate of increase of nested MC samples per level N_{0,l} = N_00 2^{gamma*l}

--alpha, --beta: Initial estimates for the decay of the bias and variance of
multilevel correction terms

--R: The adaptive refinemet parameter 1<R<2. Setting R<0 uses non-adaptive sampling

--name: A name for the folder to store data

--mode: One of `level_stats` - to estimate statistics of the multilevel differences, or `mlmc` to compute a full MLMC estimate over a range of error tolerances

--L_test: If --mode is `level_stats`, the number of levels to approximate statistics for

--t_max: Maximum runtime of the algorithm

Example commands can be cound in `commands.sh`
