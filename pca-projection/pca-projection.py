import numpy as np

def pca_projection(X: list, k: int) -> list:
    """
    Returns the centered data projected onto the top components.
    """
    x = np.array(X, dtype=float)
    mean = np.mean(x, axis=0)
    xc = x - mean
    cov = (xc.T @ xc) / (x.shape[0] - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort eigenvectors by descending eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Select top k principal components
    components = eigenvectors[:, :k]

    # Step 7: Project centered data
    projected = xc @ components

    return projected.tolist()