import numpy as np
from configParser.globalWords import activationFunct
from error_message import MyValidationError


def sigmoid(Z):
    #Z :numpy array dimension: num_samples, layer size
    # output: sigmoid(z) dimension: num_samples, layer size
    return 1 / (1 + np.exp(-Z))

def tanh(Z):
    #Z :numpy array dimension: num_samples, layer size
    # output: tanh(z) dimension: num_samples, layer size
    return np.tanh(Z)

def relu(Z):
    #Z :numpy array dimension: num_samples, layer size
    # output: relu(z) dimension: num_samples, layer size
    return np.maximum(0, Z)

def linear(Z):
    #Z :numpy array dimension: num_samples, layer size
    # output: Z dimension: num_samples, layer size
    return Z

def softmax(Z):
    # Z :numpy array dimension: num_samples, classes
    # output: softmax(z) dimension: num_samples, classes
    #loop over each sample (row)
    ris = np.zeros((Z.shape[0], Z.shape[1]))
    for i in range(Z.shape[0]):
        ris[i, :] = np.exp(Z[i,:]) / np.sum(np.exp(Z[i,:]))
    return ris

def activation_forward(function, X):
    if function == activationFunct.SIGMOID:
        return sigmoid(X)
    elif function == activationFunct.TANH:
        return tanh(X)
    elif function == activationFunct.RELU:
        return relu(X)
    elif function == activationFunct.LINEAR:
        return linear(X)
    elif function == activationFunct.SOFTMAX:
        return softmax(X)
    else:
        raise MyValidationError("Activation error")


#--------------------BACKWARD FUNCTION---------------

def tanh_backward(X):
    #Z :numpy array   output: backward of tanh(X)
    return 1 - np.power(np.tanh(X), 2)


def relu_backward(X):
    #Z :numpy array   output: backward of relu(X)
    out = np.copy(X)
    out[X > 0] = 1
    out[X <= 0] = 0
    return out

# the derivative of softmax uses the Kronecker delta
# if i == j s_i * (1 - s_i) otherwise -s_i * s_j
def softmax_grad(s):
    jacobian = np.diag(s)
    for i in range(len(jacobian)):
        for j in range(len(jacobian)):
            if i == j:
                jacobian[i][j] = s[i] * (1-s[i])
            else:
                jacobian[i][j] = -s[i]*s[j]
    return jacobian

def softmax_backward(x):
    return np.array([softmax_grad(row) for row in x])


def sigmoid_backward(X):
    #Z :numpy array   output: backward of sigmoid(X)
    return sigmoid(X) * (1 - sigmoid(X))


def linear_backward(X):
    #Z :numpy array   output: backward of linear(X)
    return np.ones(X.shape)


def activation_backward(function, X):
    if function == activationFunct.SIGMOID:
        return sigmoid_backward(X)
    elif function == activationFunct.TANH:
        return tanh_backward(X)
    elif function == activationFunct.RELU:
        return relu_backward(X)
    elif function == activationFunct.LINEAR:
        return linear_backward(X)
    elif function == activationFunct.SOFTMAX:
        return softmax_backward(X)
    else:
        raise MyValidationError("Activation error")

