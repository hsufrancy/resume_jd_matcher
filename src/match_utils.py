import numpy as np

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0)) # make sure the value is in [-1, 1]


    