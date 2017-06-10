"""
    edge Detection CPU implementation
    @philipchicco 
"""
# imports
from utils.edgeDetector_Abs import _EdgeDetector
from tools.Picture import Picture
import numpy as np
import math


class edgeDetector_cpu(_EdgeDetector):
    filter_1 = None
    filter_2 = None

    def __init__(self, filter_1=None, filter_2=None):
        """
        create filters by default, filters must be numpy arrays
        """
        if filter_1 is None:
            self.filter_1 = np.array([
                [-1, 0, 1],
                [-2, 0, 2],
                [-1, 0, 1]
            ], dtype=np.int)

        if filter_2 is None:
            self.filter_2 = np.array([
                [1, 0, 1],
                [0, 0, 0],
                [-1, -2, -1]
            ], dtype=np.int)

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
            return np.uint8(value) #grayscale value

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

    def calculateEdges(self, image, channel=1):
        """
        calculate the edges in a given image
        :param channel: 
        :param image: instance of Picture
        :return: new image with edge detection 
        """
        if isinstance(image, Picture):
            width = image.width()
            height = image.height()
            edged_image = Picture(width=width, height=height, channel=channel)

            for y in range(height):
                for x in range(width):

                    # get neighbourhood of colors 3 * 3
                    gray = np.zeros((3, 3), dtype=np.int)
                    for i in range(3):
                        for j in range(3):
                            x_value = x - 1 + i
                            y_value = y - 1 + j
                            # avoid Index error
                            #print(x_value, y_value)
                            if x_value < width and y_value < height:
                                if channel == 1:
                                    gray[i][j] = np.int(image.get(x=x_value, y=y_value, channel=1))
                                else:
                                    gray[i][j] = np.int(image.intensity(x=x_value, y=y_value))

                    # apply filters
                    gray_1 = 0
                    gray_2 = 0
                    for i in range(3):
                        for j in range(3):
                            gray_1 += gray[i][j] * self.filter_1[i][j]
                            gray_2 += gray[i][j] * self.filter_2[i][j]

                    # calculate magnitude and truncate, set format here or in picture func set
                    magnitude = 255 - self.truncate(np.int(math.sqrt(gray_1 * gray_1 + gray_2 * gray_2)))
                    # set grayscale value
                    edged_image.set(x=x, y=y, i=magnitude, channel=channel)
            print("done")
        else:
            print("Error : Image object is not an instance of Picture ")
            return
        # computation done
        return edged_image

