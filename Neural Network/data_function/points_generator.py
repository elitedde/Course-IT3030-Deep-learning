import numpy as np
from error_message import MyValidationError
import random


#points for rectangles and squares
def rect_coordinates(n, square, x0, x1, y0, y1, centered):
    length_x = random.randint(x0, x1)
    width_y = random.randint(y0, y1)
    if centered is True:
        start_x = int(n / 2) - int(length_x / 2)
        start_y = int(n / 2) - int(width_y / 2)
        if start_x < 0 or start_y < 0:
            raise MyValidationError("Error range")
    else:
        start_x = random.randint(0, n - length_x)
        start_y = random.randint(0, n - width_y)
    #check out of range matrix
    if start_x + length_x > n or start_y + width_y > n:
        raise MyValidationError("Error range")
    if square is not True:
        return start_x, length_x, start_y, width_y
    length = min(length_x, width_y)
    return start_x, length, start_y

#points for vertical and horizontal bars
def bar_coordinates(n, x0, x1, centered):  # width is fixed to 1
    length_x = random.randint(x0, x1)
    if centered is True:
        start_x = int(n / 2) - int(length_x / 2)
        if start_x < 0:
            raise MyValidationError("Error range")
        start_y = int(n / 2)
    else:
        start_x = random.randint(0, n - length_x)
        start_y = random.randint(0, n - 1)
    #check out of range matrix
    if start_x + length_x > n:
        raise MyValidationError("Error range")
    return start_x, length_x, start_y

def set_random_points(matrix, points, n):
    indices = np.random.randint(0, high=n, size=points * 2)
    # Extract the row and column indices
    i = indices[0: points]
    j = indices[points: points * 2]
    # Put 1 at the random position
    matrix[i, j] = 1

#set zeros matrix
def set_zeros_matrix(n):
    return np.zeros((n, n))

def get_number_noise_point(perc, m):
    return int(m / 100 * perc)

#zeros matrix + noise points
def set_matrix_noise_p(n, noise):
    matrix = set_zeros_matrix(n)
    points = get_number_noise_point(noise, n * n)
    set_random_points(matrix, points, n)
    return matrix