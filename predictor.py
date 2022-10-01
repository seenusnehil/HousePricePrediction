import numpy as np
import matplotlib.pyplot as plt


def predict(x, w, b):
    ans = np.dot(x, w) + b
    return ans
