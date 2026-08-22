import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.array(x)
    res = 1/(1+np.exp(-x))
    return res.item() if res.ndim == 0 else res
    pass