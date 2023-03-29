import numpy as np

# Define the sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

# Define the backpropagation function
def backpropagation(X, y, num_epochs, learning_rate):
    # Initialize the weights randomly
    num_inputs = X.shape[1]
    num_hidden = 4
    num_outputs = y.shape[1]
    W1 = np.random.randn(num_inputs, num_hidden)
    W2 = np.random.randn(num_hidden, num_outputs)

    # Loop over the number of epochs
    for epoch in range(num_epochs):
        # Forward pass
        Z1 = np.dot(X, W1)
        A1 = sigmoid(Z1)
        Z2 = np.dot(A1, W2)
        A2 = sigmoid(Z2)

        # Backward pass
        delta2 = (A2 - y) * sigmoid_derivative(Z2)
        dW2 = np.dot(A1.T, delta2)
        delta1 = np.dot(delta2, W2.T) * sigmoid_derivative(Z1)
        dW1 = np.dot(X.T, delta1)

        # Update the weights
        W1 -= learning_rate * dW1
        W2 -= learning_rate * dW2

    return W1, W2

# Define the training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Train the network using backpropagation
num_epochs = 10000
learning_rate = 0.1
W1, W2 = backpropagation(X, y, num_epochs, learning_rate)

# Use the trained network to make predictions
Z1 = np.dot(X, W1)
A1 = sigmoid(Z1)
Z2 = np.dot(A1, W2)
A2 = sigmoid(Z2)
predictions = np.round(A2).astype(int)

# Print the predictions
for i in range(len(X)):
    print(f"Input: {X[i]}, Target: {y[i]}, Prediction: {predictions[i]}")
