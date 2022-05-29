import numpy as np

from activation_functions import activation_forward, activation_backward


class Bias:
    def __init__(self, lo, hi, dim, prev):
        self.b = np.float32(np.random.uniform(lo, hi, size=(dim, ))) # bias
        self.db = None # bias gradient

class Weight:
    def __init__(self):
        self.w = None # weight
        self.dw = None # weight gradient


    def glorot_normal(self, l_next, l_previous, ):
        sd = np.sqrt(2.0 / (l_previous + l_next))
        self.w = np.float32(np.random.normal(0.0, sd, size=(l_next, l_previous)))

    def init_weights_range(self, lo, hi, l_next, l_previous):
        self.w = np.float32(np.random.uniform(lo, hi, size=(l_next, l_previous)))

    def glorot_uniform(self, l_next, l_previous):
        sd = np.sqrt(6.0 / (l_previous + l_next))
        self.w = np.float32(np.random.uniform(-sd, sd, size=(l_next, l_previous)))

# INPUT CLASS
class Input:
    def __init__(self, size_layer, i):
        self.Z = None
        self.A = None
        self.name = "Layer" + str(i)
        self.size_layer = size_layer

    def forward_function(self, X):
        self.Z = X
        self.A = X
        return self.A

    def print_layer(self):
        print(self.name + "     " + str(self.size_layer))

class HiddenLayer:

    def __init__(self, size_layer, size_layer_prev, activation_function, lrate,
                 weight, bias, i):
        self.X = None
        self.Z = None
        self.A = None
        self.activation_function= None
        self.name = "Layer" + str(i)
        self.size_layer = size_layer
        self.weight = Weight()
        if weight == "GLOROT_NORMAL":
            self.weight.glorot_normal(size_layer_prev, size_layer)
        elif weight == "GLOROT_UNIFORM":
            self.weight.glorot_uniform(size_layer_prev, size_layer)
        else:
            self.weight.init_weights_range(weight[0], weight[1], size_layer_prev, size_layer)

        self.bias = Bias(bias[0], bias[1], size_layer, 1)
        self.activation_function = activation_function
        self.lrate = lrate

    def forward_function(self, X):
        self.Z = np.dot(X, self.weight.w) + self.bias.b
        self.A = activation_forward(self.activation_function, self.Z)
        return self.A


    def backward_function(self, JLZ, a_prev):
        # da dimension = n samples, layer size
        # output dimension = n samples, layer -1 size

        # Derivative of output with respect to the sum of Jacobian
        J_diag = activation_backward(self.activation_function, self.Z)

        # Derivative of output with respect to the output of the previous layer.
        JZY = np.einsum("ij,jk->ijk", J_diag, np.transpose(self.weight.w))

        # Derivative of output with respect to  the output of this layer.
        JZW = np.einsum('ij,ik->ijk', a_prev, J_diag)

        # Derivative of loss with respect to  the weights of this layer.
        self.weight.dw = np.einsum("ij,ikj->ikj", JLZ, JZW)

        # Derivative of loss with respect to  the output of the previous layer.
        JLY = np.einsum("ij,ijk->ik", JLZ, JZY)

        # Derivative of loss w.r.t the biases of this layer.
        self.bias.db = JLZ * J_diag

        return JLY

    def print_layer(self):
        print(self.name + "   activation  " + str(self.activation_function) + "    weights  " + str(self.weight.w.shape))



class Softmax():

    def __init__(self, size_layer, activation_function, i):
        self.Z = None
        self.A = None
        self.name = "Layer" + str(i+1)
        self.size_layer = size_layer
        self.activation_function = activation_function

    def forward_function(self, X):
        self.Z = X
        self.A = activation_forward(self.activation_function, self.Z)
        return self.A

    def backward_function(self, da, a):
        dz = activation_backward(self.activation_function, self.A)
        dz = np.einsum("ij, ijk ->ik", da, dz)
        return dz

    def print_layer(self):
        print(self.name + " activation  " + str(self.activation_function))
