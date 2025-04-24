import numpy as np


np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# Sample data: XOR Logic Gate (2 inputs, 1 output)
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# XOR truth values
y = np.array([[0],
              [1],
              [1],
              [0]])


# Network architecture
input_size = 2
hidden_size = 4
output_size = 1


# Initialize weights and biases
W1 = np.random.randn(input_size, hidden_size)   # from input to hidden layer
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)   # from hidden to output layer
b2 = np.zeros((1, output_size))


# Forward pass
z1 = X @ W1 + b1
a1 = sigmoid(z1)

z2 = a1 @ W2 + b2
a2 = sigmoid(z2)

# Print forward output
print("Predicted Output:\n", a2)





# Training parameters
epochs = 10000
learning_rate = 0.1


def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-8) +
                    (1 - y_true) * np.log(1 - y_pred + 1e-8))


# Training loop
for epoch in range(epochs):
    # ---- Forward pass ----
    z1 = X @ W1 + b1
    a1 = sigmoid(z1)

    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    # ---- Loss ----
    loss = binary_cross_entropy(y, a2)

    # ---- Backpropagation ----
    dz2 = a2 - y    # derivative of loss w.r.t. z2
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1 = dz2 @ W2.T
    dz1 = da1 * sigmoid_derivative(z1)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # ---- Update weights ----
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    # ---- Print every 1000 epochs ----
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")





print("\nFinal predictions after training:")
print(a2)

# Apply threshold of 0.5 to convert to binary predictions
predicted_labels = (a2 > 0.5).astype(int)
print("\nFinal predicted labels (rounded):")
print(predicted_labels)



"""
outputs:
[[6.74722919e-04]  ≈ 0 → for input [0, 0]
 [9.98833040e-01]  ≈ 1 → for input [0, 1]
 [9.98177597e-01]  ≈ 1 → for input [1, 0]
 [2.20178419e-03]] ≈ 0 → for input [1, 1]

this matches expected XOR outputs:
X = [[0, 0],  → 0
     [0, 1],  → 1
     [1, 0],  → 1
     [1, 1]]  → 0
"""




