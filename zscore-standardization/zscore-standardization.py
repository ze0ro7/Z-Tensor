
import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Returns population Z-scores as a NumPy array matching the shape of X.
    """
    x = np.asarray(X, dtype=float)
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    z = np.zeros_like(x, dtype=float)
    np.divide( x - mean, std, out=z, where=std > eps)
    return z
