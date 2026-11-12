import numpy as np

def bootstrap_mean(x: list, n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0) -> dict:
    """
    Returns a dictionary with bootstrap_mean, lower, and upper.
    """
    x = np.array(x, dtype=float)
    # Create random number generator
    rng = np.random.default_rng(seed)
    n = len(x)
    # Generate bootstrap samples
    samples = rng.choice(x, size=(n_bootstrap, n), replace=True)
    # Calculate mean of each bootstrap sample
    means = np.mean(samples, axis=1)
    # Calculate confidence interval
    alpha = (1 - ci) / 2
    lower = np.percentile(means, alpha * 100)
    upper = np.percentile(means, (1 - alpha) * 100)
    return {
        "bootstrap_mean": float(np.mean(means)),
        "lower": float(lower),
        "upper": float(upper)
    }