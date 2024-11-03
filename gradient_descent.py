import numpy as np
import time
from cost_f import cost_f

def compute_gradient(X, Y, w, b):

    d_w = np.zeros(w.shape[0])
    d_b = 0

    n = X.shape[0]

    for idx in range(n):
        err = (np.dot(X[idx], w) + b) - Y[idx]
        for j in range(X.shape[1]):
            d_w[j] += (err * X[idx,j])
        d_b += err
    d_b /= X.shape[0]
    d_w /= X.shape[0]

    return d_w, d_b

def gradient_descent(X, Y, w_in, b_in, alpha, num_iters):

    w = w_in.copy()
    b = b_in

    for i in range(num_iters):

        wj_d, bj_d = compute_gradient(X, Y, w, b)
        w = w - alpha * wj_d
        b = b - alpha * bj_d

        if i % 100 == 0:
            print(f"Time: {time.strftime('%H:%M:%S')}; Iteration {i}; Cost {cost_f(w, b, X, Y)};")

    return w, b


