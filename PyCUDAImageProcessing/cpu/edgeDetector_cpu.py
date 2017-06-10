"""
    edge Detection CPU implementation : SOBEL ALGORITHM
    @philipchicco 
"""
# imports
from utils.edgeDetector_Abs import _EdgeDetector
from tools.Picture import Picture
import numpy as np
import math


class edgeDetector_cpu(_EdgeDetector):
    filter_1 = None  # sobel_x
    filter_2 = None  # sobel_y

    def __init__(self, filter_1=None, filter_2=None):
        """
        create filters by default, filters must be numpy arrays
        """
        if filter_1 is None:
            self.filter_1 = np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ])

        if filter_2 is None:
            self.filter_2 = np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ])

        if filter_1 is not None and filter_2 is not None:
            try:
                self.filter_1 = filter_1
                self.filter_2 = filter_2
            except np.linalg.LinAlgError as numpy_error:
                print("Invalid array given as argument: {}".format(numpy_error))
                raise

    def truncate(self, value):
        """
        :param value: a pixel value 
        :return: a truncated value of the pixel
        """
        if value < 0:
            return 0
        elif value > 255:
            return 255
        else:
            return value  # grayscale value

    def set_filter_1(self, array):
        """
        set the filter 1 values
        :param array: 
        :return: None 
        """
        if isinstance(array, np.ndarray):
            self.filter_1 = array
        else:
            print("given argument is not instance of numpy.array")
            raise

    def set_filter_2(self, array):
        """
        set the filter 2 values
        :param array: 
        :return: None
        """
        if isinstance(array, np.ndarray):
            self.filter_2 = array
        else:
            print("given argument is not instance of numpy.array")
            raise

    def get_filter_1(self):
        """
        filter 1 array
        :return: numpy array
        """
        return self.filter_1

    def get_filter_2(self):
        """
        filter 2 array
        :return: numpy array
        """
        return self.filter_2

    def sobel_edges(self, image, channel=1):
        """
        SOBEL ALGORITHM IMPLEMENTATION
        :param image: 
        :param channel: 
        :return: 
        """
        sum_x = 0
        sum_y = 0
        if isinstance(image, Picture):
            width = image.width()
            height = image.height()
            edged_image = Picture(width=width, height=height, channel=channel)

            for y in range(height):
                for x in range(width):

                    # convolution
                    mag = self.truncate(self.convolution(image.get_image_array(), x, y))
                    edged_image.set(x=x, y=y, i=mag, channel=channel)

        return edged_image
        # end if block

    def convolution(self, image, x, y):
        """
        :param image: image array 
        :param x: x-co-rdinate
        :param y: y-co-ordinate
        :return: new_pixel value
        """
        ret_x = 0
        ret_y = 0
        lx, ly = image.shape

        for i in range(-1, 2):
            for j in range(-1, 2):
                x_index = (x + i if x + i < lx else (x + i) % lx)
                y_index = (y + j if y + j < ly else (y + j) % ly)
                ret_x += image[x_index, y_index] * self.filter_1[i + 1, j + 1]
                ret_y += image[x_index, y_index] * self.filter_2[i + 1, j + 1]

        return math.sqrt(ret_x ** 2 + ret_y ** 2)