import numpy as np

# The following code returns estimates of  E{H(g_{l+eta_\ell}) - H(g_{l-1+\eta_{|ell-1}})}
def adaptive_sampler(ell, ell0, M, r, c, sampler, theta, rng):
    """
    l -> current level; l0 -> initial level
    M -> # samples
    r -> adaptive mlmc tuning parameter
    sampler -> Sampler class imported from samplers.py
    """
    sampler.init_ell(ell, ell0)  # Initialise sampler to level l
    # Initial noise for fine level:
    noise, cost = sampler.sample_noise(M, rng)
    # Initialise output parameters:
    cost_f = cost  # Cost at fine level only
    sum_diff = np.zeros(4)  # Sum of first two powers of multilevel correction terms
    fine = 0  # Sum of fine samples
    fine_sq = 0  # Sum of squared fine samples
    sum_vals = np.zeros(2)
    # Fine Level:
    eta = 0  # Adaptive correction
    while noise.size > 0:  # Loop until we no longer require new levels
        g_fine, cost_t = sampler.sample_g(noise, ell, eta, rng)  # Generate fine samples from noise
        cost += cost_t

        # Test for acceptance according to delta(g_fine) and record accepted/rejected terms:
        done = (((sampler.delta(g_fine, noise, ell, eta) < c*2**(sampler.gamma * (theta*ell*(1 - r) - eta)/r))\
            *(eta < theta*ell)) == False).astype(bool)
        noise_acc, noise = sampler.split_noise(noise, done)
        g_acc_f = g_fine[:, done == True]

        # Coarse level (if required):
        if ell > ell0:
            eta_coarse = 0  # Adapted correction for the coarse level

            # Loop over eta_coarse until the points accepted at ell + eta_ell are accepted at ell - 1 + eta_coarse:
            while noise_acc.size > 0:
                if eta_coarse > eta+1.1:  # If noise is not already refined enough, refine further
                    noise_acc, cost_t = sampler.refine_noise(noise_acc, ell - 1, eta_coarse, rng)
                    cost += cost_t

                # Use sampled noise to compute coarse sample
                g_coarse, cost_t = sampler.sample_g(noise_acc, ell-1, eta_coarse, rng)
                cost += cost_t

                # Test for acceptance according to delta(g_coarse) and divide accepted/rejected noise:
                done = (((sampler.delta(g_coarse, noise_acc, ell - 1, eta_coarse)\
                    < c*2**(sampler.gamma * (theta*(ell - 1)*(1 - r) - eta_coarse)/r))\
                    *(eta_coarse < theta*(ell-1))) == False).astype(bool)
                noise_done, noise_acc = sampler.split_noise(noise_acc, done)

                # Update terms
                correct, fine_term, cost_t, cost_f_t = sampler.multilevel_correction(g_acc_f[:, done], g_coarse[:, done], \
                    ell + eta, ell - 1 + eta_coarse, noise_done, rng)
                sum_diff += np.array([np.sum(correct**i) for i in [1,2,3,4]])
                for i in range(2):
                    sum_vals[i] += np.sum(correct*(-1)**(i+1) > 0)

                fine += np.sum(fine_term)
                fine_sq += np.sum(fine_term**2)
                g_acc_f = g_acc_f[:, done == False]
                cost += cost_t
                cost_f += cost_f_t
                eta_coarse += 1

        else:  # If coarse level is not required, record fine terms
            correct, cost_t, cost_f_t = sampler.multilevel_correction(g_acc_f, 'NA', ell + eta, 0, noise_acc, rng)
            sum_diff += np.array([np.sum(correct**i) for i in [1,2,3,4]])
            sum_vals[-1] += np.sum(correct>0)
            cost += cost_t
            cost_f += cost_f_t
            fine += sum_diff[0]
            fine_sq += sum_diff[1]

        noise, cost_t = sampler.refine_noise(noise, ell, eta, rng) # Refine rejected noise
        cost += cost_t
        cost_f += cost_t

        eta += 1 # Update adaptive correction for fine level

    return (sum_diff, sum_vals, fine, fine_sq, cost, cost_f)

def det_sampler(ell, ell0, M, sampler, rng):
    """
    Mimics the function above but samples g at deterministic levels of approximation
    """
    sampler.init_ell(ell, ell0)

    # Initialise noise:
    noise, cost = sampler.sample_noise(M, rng)
    cost_f = cost

    # Fine level
    g_fine, cost_t = sampler.sample_g(noise, ell, 0, rng)
    cost += cost_t
    cost_f += cost_t
    sum_vals = np.zeros(2)
    
    # Coarse level
    if ell > ell0:
        g_coarse, cost_t = sampler.sample_g(noise, ell-1, 0, rng)
        cost += cost_t
        correct, fine_term, cost_t, cost_f_t = sampler.multilevel_correction(g_fine, g_coarse, ell, ell-1, noise, rng)
        cost += cost_t
        cost_f += cost_f_t
        for i in range(2):
            sum_vals[i] += np.sum(correct*(-1)**(i+1) > 0)
    else:
        g_coarse = 'NA'
        correct, cost_t, cost_f_t = sampler.multilevel_correction(g_fine, g_coarse, ell, ell-1, noise, rng)
        cost += cost_t
        cost_f += cost_f_t
        fine_term = correct
        sum_vals[-1] += np.sum(correct > 0)
    return (np.array([np.sum((correct)**i) for i in [1,2,3,4]]), sum_vals, np.sum(fine_term), np.sum(fine_term**2), cost, cost_f)
