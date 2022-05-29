from enum import Enum

class regularization(Enum):
    L1 = "L1"
    L2 = "L2"
    no = "notfound"

class weightFunct(Enum):
    GLOROT_NORMAL = "GLOROT_NORMAL"
    GLOROT_UNIFORM = "GLOROT_UNIFORM"

class activationFunct(Enum):
    LINEAR = "LINEAR"
    RELU = "RELU"
    TANH = "TANH"
    SIGMOID = "SIGMOID"
    SOFTMAX = "SOFTMAX"


class lossFunct(Enum):
    MSE = "MSE"
    CROSS_ENTROPY = "CROSS_ENTROPY"

