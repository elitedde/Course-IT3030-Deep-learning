import numpy as np
from error_message import MyValidationError
import matplotlib.pyplot as plt

def plot_images(dataset, flatten):
    d = dataset
    ncols = 5
    n_images = 10
    dim = int(np.sqrt(d.shape[1]))
    #indices of the dataset are randomly selected
    indices = list(np.random.randint(0, high=d.shape[0], size=n_images))
    nrows = int(np.ceil(n_images / ncols))
    plt.subplots(nrows, ncols)
    for i in range(n_images):
        a = plt.subplot(nrows, ncols, i + 1)
        a.axis('off')
        if flatten:
            im = d[indices[i], :].reshape(dim, dim)
        else:
            im = d[indices[i], :, :]
        plt.imshow(im, cmap='gray')
    plt.show()

def plot_loss_and_accuracy(train_loss_history, train_accuracy_history, val_loss_history, val_accuracy_history,
                                                test_loss, test_accuracy, epochs):
    # x: EPOCH
    # y : LOSS
    plt.subplots(2, 1)
    plt.subplot(2, 1, 1)
    plt.plot(train_loss_history, 'r')
    plt.plot(val_loss_history, 'b')
    plt.plot(epochs - 1, test_loss, 'go')
    plt.xlabel("EPOCH")
    plt.ylabel("LOSS")
    plt.legend(('train_loss', 'val_loss', "test_loss"), loc='upper right')

    # x: EPOCH
    # y : ACCURACY
    plt.subplot(2, 1, 2)
    plt.plot(train_accuracy_history, 'r')
    plt.plot(val_accuracy_history, 'b')
    plt.plot(epochs - 1, test_accuracy, 'go')
    plt.xlabel("EPOCH")
    plt.ylabel("ACCURACY")
    plt.legend(('train_accuracy', 'val_accuracy', "test_accuracy"), loc='lower right')
    plt.show()


