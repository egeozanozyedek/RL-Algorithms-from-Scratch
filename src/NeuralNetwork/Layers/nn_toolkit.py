import numpy as np

"""
"""

def relu(X):

    phi = X * (X > 0)
    grad = 1 * (X > 0)

    return phi, grad


def silu(X):

    sigX = sigmoid(X)[0]

    phi = X * sigX
    grad = sigX * (1 + X * (1 - sigX))

    return phi, grad


def tanh(X):

    phi = np.tanh(X)
    grad = 1 - phi ** 2

    return phi, grad


def sigmoid(X):

    phi = 1 / (1 + np.exp(-X))
    grad = phi * (1 - phi)

    return phi, grad


def linear(X):

    phi = X
    grad = 1

    return phi, grad


def cross_entropy(pred, actual):


    error = np.sum(- actual * np.log(pred)) / len(actual)

    residual = pred
    residual[actual == 1] -= 1
    residual /= len(actual)

    return error, residual



def mse(pred, actual):

    # print("Pred:", pred.shape, pred.min(), pred.max(), "Actual:", actual.shape, actual.min(), actual.max())

    if isinstance(actual, float):
        size = 1
    else:
        size = len(actual)

    error = ((actual - pred)**2).mean()

    residual = (pred - actual)/size


    return error, residual




activation_map = {"relu": relu, "silu": silu , "tanh": tanh, "sigmoid": sigmoid, "linear": linear}
error_map = {"MSE": mse, "CE": cross_entropy}
