import numpy as np

def calculate_eigenvalues(matrix: list) -> np.ndarray:
    """
    Returns a sorted NumPy array of real eigenvalues.
    """
    # Write code here
    x = np.array(matrix, dtype = float)
    eigenval = np.linalg.eigvals(x)
    return np.sort(np.real(eigenval))