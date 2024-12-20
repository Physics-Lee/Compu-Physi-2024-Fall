import numpy as np
import matplotlib.pyplot as plt

def monte_carlo_integration(func, a, b, n_samples, true_value, use_seed = False):
    """
    Basic Monte Carlo integration to estimate the integral of a given function `func`
    over the interval [a, b] using `n_samples` random samples.
    """
    if use_seed:
        np.random.seed(42)
    x = np.random.uniform(a, b, n_samples)  # Random samples in [a, b]
    fx = func(x)  # Evaluate the function at sampled points
    estimate = (b - a) * np.mean(fx)  # Estimate of the integral
    abs_error = np.abs(estimate - true_value)  # Absolute error
    return estimate, abs_error

def stratified_sampling(func, a, b, n_samples, strata, true_value, use_seed = False):
    """
    Monte Carlo integration using stratified sampling. 
    The interval [a, b] is divided into 'strata' subintervals.
    """
    if use_seed:
        np.random.seed(42)
    samples_per_stratum = n_samples // strata
    estimates = [
        (b - a) / strata * np.mean(func(np.random.uniform(a + i * (b - a) / strata,
                                                          a + (i + 1) * (b - a) / strata, 
                                                          samples_per_stratum)))
        for i in range(strata)
    ]
    estimate = sum(estimates)
    abs_error = np.abs(estimate - true_value)
    return estimate, abs_error

def importance_sampling(func, a, b, n_samples, true_value = None, importance_dist = None, use_seed = False):
    """
    Monte Carlo integration using importance sampling.
    The probability distribution for sampling is given by 'importance_dist'.
    """
    if use_seed:
        np.random.seed(42)
    
    if importance_dist is None:
        raise ValueError("An importance sampling distribution must be provided.")
    
    # Generate a large batch of samples and filter them to be within [0, 4]
    x = importance_dist['sample'](int(n_samples * 1.5))  # Generate more samples initially
    x = x[(x >= a) & (x <= b)]  # Filter samples to [0, 4]
    
    # If not enough samples, repeat the process
    while len(x) < n_samples:
        extra_samples = importance_dist['sample'](int(n_samples * 0.5))
        extra_samples = extra_samples[(extra_samples >= a) & (extra_samples <= b)]
        x = np.concatenate((x, extra_samples))
    
    # Use exactly the first n_samples samples after filtering
    x = x[:n_samples]
    fx = func(x) / importance_dist['pdf'](x)  # Weighted function evaluation
    estimate = np.mean(fx)
    abs_error = np.abs(estimate - true_value) if true_value is not None else None
    
    # # Plot the histogram of x
    # plt.hist(x, bins=30, alpha=0.7, color='b', edgecolor='black')
    # plt.title('Histogram of Sampled Values of Importance Sampling')
    # plt.xlabel('x')
    # plt.ylabel('Frequency')
    # plt.show()

    return estimate, abs_error

def metropolis_sampling_with_pbc(n_samples, g, a, b, proposal_std=0.5, init_x=2.0, use_seed=False):
    """
    Metropolis algorithm for sampling from a target distribution g(x) on the interval [a, b] with PBC.
    
    Parameters:
        n_samples (int): Number of samples to generate.
        g (function): Target distribution g(x) proportional to the density.
        a (float): Lower bound of the interval.
        b (float): Upper bound of the interval.
        proposal_std (float): Standard deviation for the Gaussian proposal distribution.
        init_x (float): Initial value for the Markov chain.
    
    Returns:
        samples (array): Array of sampled values.
        acceptance_rate (float): The rate of accepted proposals.
    """
    if use_seed:
        np.random.seed(42)
    samples = []
    current_x = init_x
    accepted = 0
    
    for _ in range(n_samples):
        # Proposal step: Gaussian proposal centered at current_x
        proposal_x = np.random.normal(current_x, proposal_std)
        
        # Apply PBC: wrap proposal_x within [a, b]
        if proposal_x < a:
            proposal_x = b - (a - proposal_x) % (b - a)
        elif proposal_x > b:
            proposal_x = a + (proposal_x - b) % (b - a)
        
        # Calculate acceptance ratio
        acceptance_ratio = min(1, g(proposal_x) / g(current_x))
        
        # Accept or reject the proposal
        if np.random.rand() < acceptance_ratio:
            current_x = proposal_x  # Accept the proposal
            accepted += 1
        
        samples.append(current_x)
    
    # Calculate acceptance rate
    acceptance_rate = accepted / n_samples
    
    return np.array(samples), acceptance_rate

def compare(n_samples_list, n_exp, f, a, b, true_value, importance_dist):
    """
    Compare the performance of Monte Carlo, Stratified Sampling, and Importance Sampling
    by running multiple experiments with different number of samples.
    """

    # Initialize arrays to store absolute errors
    mc_errors = np.zeros((len(n_samples_list), n_exp))
    stratified_errors = np.zeros((len(n_samples_list), n_exp))
    importance_errors = np.zeros((len(n_samples_list), n_exp))

    # Run multiple experiments
    for i, n_samples in enumerate(n_samples_list):
        
        for j in range(n_exp):
            # Basic Monte Carlo Integration
            mc_estimate, mc_error = monte_carlo_integration(f, a, b, n_samples, true_value)
            mc_errors[i, j] = mc_error
            
            # Stratified Sampling Integration
            strat_estimate, strat_error = stratified_sampling(f, a, b, n_samples, strata = 100, true_value=true_value)
            stratified_errors[i, j] = strat_error
            
            # Importance Sampling Integration
            imp_estimate, imp_error = importance_sampling(f, a, b, n_samples, true_value, importance_dist)
            importance_errors[i, j] = imp_error

    # Calculate mean absolute errors
    mc_mean = np.mean(mc_errors, axis=1)
    stratified_mean = np.mean(stratified_errors, axis=1)
    importance_mean = np.mean(importance_errors, axis=1)

    # Calculate standard error
    mc_se = np.std(mc_errors, axis=1) / np.sqrt(n_exp)
    stratified_se = np.std(stratified_errors, axis=1) / np.sqrt(n_exp)
    importance_se = np.std(importance_errors, axis=1) / np.sqrt(n_exp)

    return mc_mean, stratified_mean, importance_mean, mc_se, stratified_se, importance_se