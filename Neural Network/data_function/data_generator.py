from configParser.configParser_data import Data
from data_function.points_generator import *


class Vertical_bar:
    def create_verticalbar(self, matrix, n, x0, x1, centered):
        start_x, length_x, start_y = bar_coordinates(n, x0, x1, centered)
        matrix[start_x:start_x + length_x, start_y] = 1


class Horizontal_bar:
    def create_horizontalbar(self, matrix, n, x0, x1, centered):
        start_x, length_x, start_y = bar_coordinates(n, x0, x1, centered)
        matrix[start_y, start_x:start_x + length_x] = 1


class Rectangle:
    def create_rectangle(self, matrix, n, x0, x1, y0, y1, centered):
        start_x, length_x, start_y, width_y = rect_coordinates(n, False, x0, x1, y0, y1, centered)
        matrix[start_x:start_x + length_x, start_y:start_y + width_y] = 1


class Square:
    def create_square(self, matrix, n, x0, x1, centered):
        start_x, length, start_y = rect_coordinates(n, True, x0, x1, x0, x1, centered)
        matrix[start_x:start_x + length, start_y:start_y + length] = 1

class DataGenerator:

    def __init__(self, args):
        self.args = args


    def dataset_generation(self, n_samples_class, n, train, val, noise, flattening,
                           rect_height, rect_width, vbar_width, hbar_width, square_range, centered):
        v_bar = Vertical_bar()
        h_bar = Horizontal_bar()
        squaree = Square()
        rectang = Rectangle()
        dataset = []
        #check out of range dimension
        if n < 10 or n > 50:
            raise MyValidationError("Error range")
        #target values generation
        Y = np.array([0] * n_samples_class + [1] * n_samples_class + [2] * n_samples_class + [3] * n_samples_class)
        one_hot_labels = np.zeros((n_samples_class * 4, 4))
        for i in range(n_samples_class * 4):
            one_hot_labels[i, Y[i]] = 1

        #rectangle set
        for i in range(n_samples_class):
            matrix = set_matrix_noise_p(n, noise)
            rectang.create_rectangle(matrix, n, rect_height[0], rect_height[1], rect_width[0], rect_width[1], centered)
            dataset.append(matrix)
        #square set
        for i in range(n_samples_class):
            matrix = set_matrix_noise_p(n, noise)
            squaree.create_square(matrix, n, square_range[0], square_range[1], centered)
            dataset.append(matrix)
        #vertical bar set
        for i in range(n_samples_class):
            matrix = set_matrix_noise_p(n, noise)
            v_bar.create_verticalbar(matrix,n, vbar_width[0], vbar_width[1], centered)
            dataset.append(matrix)
        #horizontal bar set
        for i in range(n_samples_class):
            matrix = set_matrix_noise_p(n, noise)
            h_bar.create_horizontalbar(matrix,n, hbar_width[0], hbar_width[1], centered)
            dataset.append(matrix)

        if (flattening):
            dataset = [x.flatten() for x in dataset]

        c = list(zip(dataset, one_hot_labels))
        random.shuffle(c)
        dataset, one_hot_labels = zip(*c)
        dataset, one_hot_labels = np.stack(dataset, axis=0), np.stack(one_hot_labels, axis=0)
        n = dataset.shape[0]
        train_index = int(n * train)
        val_index = train_index + int(n * val)
        return (dataset[:train_index], one_hot_labels[:train_index],
                dataset[train_index:val_index], one_hot_labels[train_index:val_index],
                dataset[val_index:], one_hot_labels[val_index:], flattening)



def manage_dataset():
        parameters = Data()
        parameters.extractData()
        dg = DataGenerator(parameters)
        return dg.dataset_generation(dg.args.class_dimension, dg.args.n, dg.args.data_train_dim, dg.args.data_val_dim,
                                     dg.args.noise_ratio, dg.args.flatten, dg.args.rect_range_height, dg.args.rect_range_width, dg.args.vertical_bar_width,
                                     dg.args.horizontal_bar_width, dg.args.square_range, dg.args.centered)


def random_mini_batches(X, Y, mini_batch_size, i):
    #the random minibatches are created
    minibatch_X= X[i:i + mini_batch_size]
    minibatch_Y = Y[i:i + mini_batch_size]
    return minibatch_X, minibatch_Y


