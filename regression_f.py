import numpy as np

def regression_function(x, w, b):
    return np.dot(x, w) + b

# f(x1, x2) = w1*x1 + w2*x2 + b