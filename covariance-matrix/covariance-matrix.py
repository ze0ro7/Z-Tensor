import numpy as np

def covariance_matrix(X: list) -> np.ndarray:
    """
    Returns the covariance matrix as a NumPy array.
    """
    # Write code here
    x = np.array(X, dtype = float)
    mean = np.mean(x, axis = 0)
    xc = x-mean
    cov = (xc.T@xc)/(x.shape[0]-1)
    return cov