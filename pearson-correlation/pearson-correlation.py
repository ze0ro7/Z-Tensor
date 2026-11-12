import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the correlation matrix as a NumPy array.
    """
    # Write code here
    x = np.array(X, dtype = float)
    mean = np.mean(x, axis = 0)
    xc = x-mean
    cov = (xc.T @ xc)/(x.shape[0]-1)
    std = np.std(x, axis = 0, ddof = 1)
    corr = cov/np.outer(std,std)
    return corr