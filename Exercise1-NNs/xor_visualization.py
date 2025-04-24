import numpy as np
import matplotlib.pyplot as plt

"""
XOR Problem Visualization:
This script visualizes the XOR problem, which is not linearly separable.
It plots the input points and their corresponding labels.

(0,0) -> 0
(0,1) -> 1
(1,0) -> 1
(1,1) -> 0
"""

# XOR input
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# XOR labels
y = np.array([0,
              1,
              1,
              0])

# Plotting the points
plt.figure(figsize=(6, 6))
for i, label in enumerate(y):
    if label == 0:
        plt.scatter(X[i, 0], X[i, 1], color='red', label='Class 0' if i == 0 else "")
    else:
        plt.scatter(X[i, 0], X[i, 1], color='blue', label='Class 1' if i == 1 else "")

# Add axis labels and grid
plt.title("XOR Problem - No Linear Decision Boundary")
plt.xlabel("Input A")
plt.ylabel("Input B")
plt.grid(True)
plt.legend()
plt.xlim(-0.2, 1.2)
plt.ylim(-0.2, 1.2)
plt.gca().set_aspect('equal', adjustable='box')

plt.show()