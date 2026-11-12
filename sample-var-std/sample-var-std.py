import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns a dictionary with variance and standard_deviation.
    """
    # Write code here
    # ddof means Delta Degrees of Freedom.
    # ddof=0 calculates population variance (divides by n).
    # ddof=1 calculates sample variance (divides by n-1).
    variance = float(np.var(x, ddof = 1))
    stdD = float(np.std(x,ddof = 1))
    return {"variance": variance, "standard_deviation": stdD}