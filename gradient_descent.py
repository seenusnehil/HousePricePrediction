import numpy as np
import matplotlib.pyplot as plt
from predictor import predict


def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0

    for i in range(m):
        f_wb_i = predict(x[i], w, b)
        err = f_wb_i - y[i]
        for j in range(3):
            dj_dw_i = err * x[i, j]
            dj_dw += dj_dw_i
        dj_db = err

    return dj_dw, dj_db


def gradient_descent(x, y, w_in, b_in, alpha, iterations):
    w = w_in
    b = b_in
    for i in range(iterations):
        dj_dw, dj_db = compute_gradient(x, y, w, b)

        w = w - alpha*dj_dw
        b = b - alpha*dj_db

    return w, b

