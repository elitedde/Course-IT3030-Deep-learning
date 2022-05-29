import numpy as np
from configParser.globalWords import activationFunct, regularization
from layer import HiddenLayer, Softmax, Input
from cost_functions import compute_cost, compute_cost_derivative, compute_regularization_der, compute_reg
from data_function.data_generator import random_mini_batches
class NN:

    def __init__(self, lrate, input_size, size_layers, activation_functions, softmax,
                 loss, regularization, regularization_rate,
                 weights, bias):
        self.layers = []
        self.input_size = input_size
        self.lrate = lrate
        self.softmax = softmax
        self.loss = loss
        self.regularization = regularization
        self.regularization_rate = regularization_rate
        self.weights = weights
        self.bias = bias
        self.size_layers = size_layers
        self.activation_functions = activation_functions
        if softmax:
            activation_functions.append(activationFunct.SOFTMAX)



    def build_NN(self):
        l = Input(self.input_size, 0)
        l.print_layer()
        self.layers.append(l)
        prec = self.input_size
        for i in range(0, len(self.size_layers)):
            l = HiddenLayer(self.size_layers[i], prec, self.activation_functions[i],
                            self.lrate[i], self.weights[i], self.bias[i], i+1)
            self.layers.append(l)
            l.print_layer()
            prec = self.size_layers[i]
        if self.softmax:
            s = Softmax(prec, activationFunct.SOFTMAX, len(self.size_layers))
            s.print_layer()
            self.layers.append(s)

    # forward function
    def NN_forward(self, X):
        for layer in self.layers:
            X = layer.forward_function(X)
        return X

    # backward function
    def NN_backward(self, output, y):
        # Calculating the first of the Jacobians
        J = compute_cost_derivative(output, y, self.loss)
        for i in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[i]
            # here i extract the previous A
            a_prev = self.layers[i-1].A
            J = layer.backward_function(J, a_prev)


    def updateParameters(self):
        for i in range(1, len(self.size_layers) + 1):
            layer = self.layers[i]
            # averaged weight gradient to obtain the same shape of w
            dw = 1 / layer.weight.dw.shape[0] * np.sum(layer.weight.dw, axis=0)
            # regularization term is added
            if self.regularization != regularization.no:
                dw += compute_regularization_der(layer.weight.w, self.regularization_rate, self.regularization)
            # update w
            layer.weight.w = layer.weight.w - layer.lrate * dw

            # averaged bias gradient to obtain the same shape of b
            db = 1 / layer.bias.db.shape[0] * np.sum(layer.bias.db, axis=0)
            # regularization term is added
            if self.regularization != regularization.no:
                db += compute_regularization_der(layer.bias.b, self.regularization_rate, self.regularization)
            # update b
            layer.bias.b = layer.bias.b - layer.lrate * db


    def model(self, train_data, targets, val_data, val_targets, batch_size, epochs=50, verbosity=1):
        # train_data: training data targets: labels for training data --> [samples, features]
        # train_data: validation data val_targets: labels for validation data --> [samples, features]
        # lists of loss and accuracy for training and validation data
        train_loss_history = []
        train_accuracy_history = []
        val_loss_history = []
        val_accuracy_history = []
        for epoch in range(epochs):
            print("Epoch {epoch}/{epochs}".format(epoch=epoch + 1, epochs=epochs))
            training_loss = 0
            # Select a minibatch
            for i in range(0, train_data.shape[0], batch_size):
                mini_batch, mini_batch_targets = random_mini_batches(train_data, targets, batch_size, i)
                output = self.NN_forward(mini_batch)
                #compute cost for the batch
                loss = compute_cost(output, mini_batch_targets, self.loss)
                training_loss += loss
                self.NN_backward(output, mini_batch_targets)
                self.updateParameters()
                if verbosity == 2:
                    print("Mini batch:")
                    print(mini_batch)
                    print("Mini batch targets:")
                    print(mini_batch_targets)
                    print("Outputs:")
                    print(output)
                    print("Mini batch loss: {batch_loss}".format(batch_loss=loss))
            # averaged batches loss
            #regularization term is added if necessary
            training_loss = training_loss / train_data.shape[0] + compute_reg(self.layers, self.regularization_rate, self.regularization)
            train_loss_history.append(training_loss)
            #-----validation set-----
            output = self.NN_forward(val_data)
            cost = compute_cost(output, val_targets, self.loss)
            # averaged batches loss
            #regularization term is added if necessary
            validation_loss = cost / val_data.shape[0] + compute_reg(self.layers, self.regularization_rate, self.regularization)
            val_loss_history.append(validation_loss)

            #compute accuracy for both training data and validation data
            accuracy = self.accuracy(train_data, targets)
            train_accuracy_history.append(accuracy)
            accuracy = self.accuracy(val_data, val_targets)
            val_accuracy_history.append(accuracy)
            if verbosity == 1:
                print("Training loss: {training_loss}".format(training_loss=training_loss))
                print("Training accuracy: {training_accuracy}".format(training_accuracy=accuracy))
                print("Validation loss: {validation_loss}".format(validation_loss=validation_loss))
                print("Validation accuracy: {validation_accuracy}".format(validation_accuracy=accuracy))

        return train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history

    def accuracy(self, X, Y):
        output = self.NN_forward(X)
        #regression problem
        if self.loss == "MSE":
            return 1 - (1 / output.shape[0]) * np.sum(np.abs(output - Y))
        #classificatio problem
        prediction = np.argmax(output, axis=1)
        a_class = np.argmax(Y, axis=1)
        accuracy = 0
        for i in range(len(prediction)):
            if prediction[i] == a_class[i]:
                accuracy += 1
        return accuracy / len(prediction)

    #test data evaluation
    def evaluate(self, test_data, test_targets):
        output = self.NN_forward(test_data)
        test_loss = compute_cost(output, test_targets, self.loss)
        return test_loss / len(test_targets) + compute_reg(self.layers, self.regularization_rate, self.regularization)

