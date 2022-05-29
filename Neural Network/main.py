from configParser.configParser_parameters import *
from neural_network import NN
from data_function.data_generator import manage_dataset
from plots import plot_loss_and_accuracy, plot_images


def main():

    x_train, y_train, x_val, y_val, x_test, y_test, flatten = manage_dataset()
    print("---------The dataset has been generated correctly------------")
    print("x train set shape: {shape}".format(shape=x_train.shape))
    print("y train shape: {shape}".format(shape=y_train.shape))
    print("x val shape: {shape}".format(shape=x_val.shape))
    print("y val set shape: {shape}".format(shape=y_val.shape))
    #plot_images(x_train, flatten)
    p = Parameters()
    p.extractParameters()
    # NN works only with a flatten dataset
    nn = NN(p.lrate, p.input_size, p.size_layer, p.activation_functions, p.softmax, p.loss, p.regularization, p.regularizationRate, p.wreg, p.br)

    nn.build_NN()
    print("---------Successfully build NN-----------------")

    train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history = nn.model(x_train, y_train, x_val, y_val, p.mini_batch_size, p.epochs, p.verbose)
    print("---------Successfully fit NN-----------------")

    test_accuracy = nn.accuracy(x_test, y_test)
    test_loss = nn.evaluate(x_test, y_test)
    print("Test loss: {test_loss}".format(test_loss=test_loss))
    print("Test accuracy: {test_accuracy}".format(test_accuracy=test_accuracy))
    print("---------Successfully calculated test loss and test accuracy-----------------")

    plot_loss_and_accuracy(train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history, test_loss, test_accuracy, p.epochs)
    print("---------Successfully plot loss and accuracy-----------------")



if __name__ == "__main__":
    main()
