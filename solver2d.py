import numpy as np

def centers(N, L):
    h = L / N
    return (np.arange(N) + 0.5) * h
