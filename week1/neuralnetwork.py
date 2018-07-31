# Basic Neural Network following this tutorial https://iamtrask.github.io/2015/07/12/basic-python-network/
import numpy as np

# sigmoid function to converts numbers to probabilities (output is [0, 1])
def sigmoid (X, w):
    return 1/(1+np.exp(-(np.dot(X, w))))

X = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]])
y = np.array([[0,1,1,0]]).T

np.random.seed(1)
w0 = 2 * np.random.random((3, 4)) - 1
w1 = 2 * np.random.random((4, 1)) - 1

for i in range(6000):
    # forward propagation
    l0 = sigmoid(X, w0)
    l1 = sigmoid(l0, w1)
    # calculate the errors of the layers using back propagation
    l1_error = (y - l1)
    l0_error = l1_error.dot(w1.T)
    # Multiply the error and the derivative of the sigmoid
    # (= "how sure it was multiplied by how wrong it was")
    l1_delta = l1_error * (l1 * (1 - l1))
    l0_delta = l0_error * (l0 * (1 - l0))
    if i % 1000 is 0:
        print ("Error", l1_delta, "\n")
    # update the weights
    w1 += l0.T.dot(l1_delta)
    w0 += X.T.dot(l0_delta)

print ("Output", l1)
