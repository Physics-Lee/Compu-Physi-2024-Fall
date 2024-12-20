import numpy as np

class MonteCarloIntegrator:
    def __init__(self, func, a, b, n_samples, true_value=None, use_seed=False):
        self.func = func
        self.a = a
        self.b = b
        self.n_samples = n_samples
        self.true_value = true_value
        self.use_seed = use_seed

    def basic_mc(self):
        # Basic Monte Carlo Integration Method
        if self.use_seed:
            np.random.seed(42)
        x = np.random.uniform(self.a, self.b, self.n_samples)
        fx = self.func(x)
        estimate = (self.b - self.a) * np.mean(fx)
        abs_error = np.abs(estimate - self.true_value) if self.true_value is not None else None
        return estimate, abs_error

    def stratified_sampling(self, strata):
        # Stratified Sampling Method
        if self.use_seed:
            np.random.seed(42)
        samples_per_stratum = self.n_samples // strata
        estimates = [
            (self.b - self.a) / strata * np.mean(self.func(np.random.uniform(
                self.a + i * (self.b - self.a) / strata,
                self.a + (i + 1) * (self.b - self.a) / strata,
                samples_per_stratum)))
            for i in range(strata)
        ]
        estimate = sum(estimates)
        abs_error = np.abs(estimate - self.true_value) if self.true_value is not None else None
        return estimate, abs_error

    def importance_sampling(self, importance_dist):
        # Importance Sampling Method
        if self.use_seed:
            np.random.seed(42)

        if importance_dist is None:
            raise ValueError("An importance sampling distribution must be provided.")

        x = importance_dist['sample'](int(self.n_samples * 1.5))
        x = x[(x >= self.a) & (x <= self.b)]

        while len(x) < self.n_samples:
            extra_samples = importance_dist['sample'](int(self.n_samples * 0.5))
            extra_samples = extra_samples[(extra_samples >= self.a) & (self.b >= extra_samples)]
            x = np.concatenate((x, extra_samples))

        x = x[:self.n_samples]
        fx = self.func(x) / importance_dist['pdf'](x)
        estimate = np.mean(fx)
        abs_error = np.abs(estimate - self.true_value) if self.true_value is not None else None
        return estimate, abs_error

    def metropolis_sampling_with_pbc(self, g, proposal_std=0.5, init_x=2.0):
        """
        Metropolis algorithm for sampling from a target distribution g(x) on the interval [a, b] with PBC.
        """
        if self.use_seed:
            np.random.seed(42)
        samples = []
        current_x = init_x
        accepted = 0
        
        for _ in range(self.n_samples):
            # Proposal step: Gaussian proposal centered at current_x
            proposal_x = np.random.normal(current_x, proposal_std)
            
            # Apply PBC: wrap proposal_x within [a, b]
            if proposal_x < self.a:
                proposal_x = self.b - (self.a - proposal_x) % (self.b - self.a)
            elif proposal_x > self.b:
                proposal_x = self.a + (proposal_x - self.b) % (self.b - self.a)
            
            # Calculate acceptance ratio
            acceptance_ratio = min(1, g(proposal_x) / g(current_x))
            
            # Accept or reject the proposal
            if np.random.rand() < acceptance_ratio:
                current_x = proposal_x  # Accept the proposal
                accepted += 1
            
            samples.append(current_x)
        
        # Calculate acceptance rate
        acceptance_rate = accepted / self.n_samples
        
        return np.array(samples), acceptance_rate

    def compare(self, n_samples_list, n_exp, importance_dist):
        """
        Compare the performance of Monte Carlo, Stratified Sampling, and Importance Sampling
        by running multiple experiments with different number of samples.
        """
        # Initialize arrays to store absolute errors
        mc_errors = np.zeros((len(n_samples_list), n_exp))
        stratified_errors = np.zeros((len(n_samples_list), n_exp))
        importance_errors = np.zeros((len(n_samples_list), n_exp))

        # 
        self.use_seed = False

        # Run multiple experiments
        for i, n_samples in enumerate(n_samples_list):

            # Update sample size for each experiment
            self.n_samples = n_samples

            for j in range(n_exp):
                
                # Basic Monte Carlo Integration
                mc_estimate, mc_error = self.basic_mc()
                mc_errors[i, j] = mc_error
                
                # Stratified Sampling Integration
                strat_estimate, strat_error = self.stratified_sampling(strata=100)
                stratified_errors[i, j] = strat_error
                
                # Importance Sampling Integration
                imp_estimate, imp_error = self.importance_sampling(importance_dist)
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