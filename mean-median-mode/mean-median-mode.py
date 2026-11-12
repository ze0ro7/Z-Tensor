from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns a dictionary with mean, median, and mode.
    """
    # Write code here
    mean = float(np.mean(x))
    median = float(np.median(x))
    counts = Counter(x)
    mode = float(min(counts, key=lambda k: (-counts[k], k)))
    return {"mean": mean, "median": median, "mode": mode}