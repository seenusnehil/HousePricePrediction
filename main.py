# Welcome to the house price predictor project
import numpy as np
import matplotlib.pyplot as plt
from predictor import predict
from cost import compute_cost
from gradient_descent import compute_gradient

x_train = np.array([[300, 2, 4], [700, 3, 6], [1640, 2, 3]])
y_train = np.array([460, 800, 1200])

weight = np.array([1, 5, 3])
bias = 10

cost = compute_cost(x_train, y_train, weight, bias)
print(cost)

dw, db = compute_gradient(x_train, y_train, weight, bias)
print(dw, db)

name = input("Enter your name: ")
print("##### Welcome to the HousePred, " + name + " #####")

size = input("House size: ")
floors = input("No. of floors: ")
rooms = input("No. of bedrooms: ")

print("Your estimated house price is: ")
