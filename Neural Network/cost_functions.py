import numpy as np
from configParser.globalWords import lossFunct
from error_message import MyValidationError

def MSE(Y, AL):
    # AL: predicted vector Y: true "label" vector
    # Returns: mse value
    return np.square(np.subtract(Y, AL)).mean()

def MSE_derivative(Y, AL):
    # AL: predicted vector Y: true "label" vector
    # Returns: mse derivative
    return AL - Y

def cross_entropy(Y, AL):
    # AL: predicted vector Y: true "label" vector
    # Returns: error number
    cr = sum(-np.sum(Y * np.log(AL), axis=1))
    return cr

def cross_entropy_derivative(Y, AL):
    # AL: predicted vector Y: true "label" vector dim:  samples, layer_size
    # Returns: cross_entropy derivative dim:  samples, layer_size
    cr = np.where(AL != 0, -Y / AL, 0.0)
    return cr

def compute_cost(AL, Y, loss):
    if loss == lossFunct.CROSS_ENTROPY:
        cost = cross_entropy(Y, AL)
    elif loss == lossFunct.MSE:
        cost = MSE(Y, AL)
    else:
        # check parameter error
        raise MyValidationError("Error parameter")
    return cost

def compute_cost_derivative(AL, Y, loss):
    if loss == lossFunct.CROSS_ENTROPY:
        cost = cross_entropy_derivative(Y, AL)
    elif loss == lossFunct.MSE:
        cost = MSE_derivative(Y, AL)
    else:
        #check parameter error
        raise MyValidationError("Error parameter")
    return cost


def l1(lr, layers):
    # L1 regularization.
    # Input: list of layers and regularization rate
    # Returns: regularization
    tot = 0
    for i in range(1, len(layers) - 1):
        layer = layers[i]
        tot += lr * np.sum(np.absolute(layer.weight.w)) + lr * np.sum(np.absolute(layer.bias.b))
    return tot

def l2(lr, layers):
    # L2 regularization.
    # Input: list of layers and regularization rate
    # Returns: regularization
    tot = 0
    for i in range(1, len(layers)-1):
        layer = layers[i]
        tot += lr * 1 / 2 * np.sum(layer.weight.w ** 2) + lr * 1 / 2 * np.sum(layer.bias.b ** 2)
    return tot

def l1_der(weight, lr):
    # L1 derivative
    # Input: weight of a layer and regularization rate
    # Returns: regularization derivative
    return lr * np.sign(weight)

def l2_der(weight, lr):
    # L2 derivative
    # Input: weight of a layer and regularization rate
    # Returns: regularization derivative
    return lr * np.array(weight)

def compute_regularization_der(weight, regularization_rate, regularization):
    if regularization == regularization.L1:
        return l1_der(weight, regularization_rate)
    elif regularization == regularization.L2:
        return l2_der(weight, regularization_rate)
    else:
        raise MyValidationError("Error parameter")


def compute_reg(layers, regularization_rate, regularization):
    if regularization == regularization.no:
        return 0
    elif regularization == regularization.L1:
        return l1(regularization_rate, layers)
    elif regularization == regularization.L2:
        return l2(regularization_rate, layers)
    else:
        raise MyValidationError("Error parameter")


