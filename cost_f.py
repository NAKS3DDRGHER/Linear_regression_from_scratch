import pandas as pd
import numpy as np
from regression_f import regression_function

def cost_f(w, b, X, Y):

    n = X.shape[0]
    cost_sum = 0

    for idx in range(n):
        f = regression_function(X[idx], w, b)
        cost = (f - Y[idx]) ** 2
        cost_sum += cost

    return cost_sum / (n * 2)


def cost_in_percent(w, b, X, Y):
    n = X.shape[0]

    cost_percent = 0

    for idx in range(n):
        f = regression_function(X[idx], w, b)
        percent = min(f, Y[idx]) / max(f, Y[idx])
        cost_percent += percent

    return cost_percent / n


