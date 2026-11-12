import numpy as np

def matrix_normalization(matrix: list, axis=None, norm_type: str = "l2") -> np.ndarray:
    """
    Returns a NumPy array with the same shape as matrix.
    """
    # Write code here
    x = np.array(matrix, dtype = float)
    if norm_type == "l1":
        norm = np.sum(np.abs(x), axis = axis, keepdims = True)
    elif norm_type == "l2":
        norm = np.sqrt(np.sum(x**2, axis = axis, keepdims = True))
    elif norm_type == "max":
        norm = np.max(np.abs(x), axis = axis, keepdims = True)
    else:
        raise ValueError("Invalid norm_type. use l1, l2 or max")
    norm = np.where(norm == 0, 1, norm)
    return x/norm