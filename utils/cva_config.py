from argparse import ArgumentParser
import numpy as np
parser = ArgumentParser()


# MLMC Convergence Parameters (Initial Estimates)
parser.add_argument('--beta', default = 1, type = np.float64)
parser.add_argument('--alpha', default=1, type = np.float64)

# (Adaptive) MLMC Parameters
parser.add_argument('--gamma', default = 1, type = int)
parser.add_argument('--R', default = -1, type = np.float64)
parser.add_argument('--N_00', default = 8, type =  int)  # Number MC samples for U_0

# Experimental set-up
parser.add_argument('--name', default = 'CVA_PLL', type = str)
parser.add_argument('--mode', default = 'level_stats', type = str)
parser.add_argument('--L_test', default = 5, type = int) # Number of Levels to test MLMC convergence
parser.add_argument('--t_max', default = 600, type=int)  # Maximum time limit of program (seconds)
