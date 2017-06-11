"""
    edge Detection implementation Class
    
"""

# imports
from utils.edgeDetector_Abs import _EdgeDetector
from tools.Picture import Picture
import numpy as np
import math


class EdgeDetector(_EdgeDetector):
    filter_1 = np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ])  # sobel_x
    filter_2 = np.array([
                [-1, -2, -1],
                [0, 0, 0],
                [1, 2, 1]
            ])  # sobel_y

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
