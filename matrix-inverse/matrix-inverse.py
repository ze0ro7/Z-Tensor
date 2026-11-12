import numpy as np

def matrix_inverse(A: list) -> np.ndarray | None:
    """
    Returns the inverse as a NumPy array, or None.
    """
    # Write code here
    x = np.array(A, dtype = float)
    try:
        return np.linalg.inv(x)
    except np.linalg.LinAlgError:
        return None