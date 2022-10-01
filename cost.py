import numpy as np
import matplotlib.pyplot as plt
from predictor import predict


# method to find the cost i.e., the difference between the predicted and the actual price
def compute_cost(x, y, w, b):
    m = x.shape[0]
    total_err = 0
    for i in range(m):
        f_wb_i = predict(x[i], w, b)
        print(f_wb_i)
        err = (f_wb_i - y[i]) ** 2
        total_err += err

    total_err /= 2 * m
    return total_err
