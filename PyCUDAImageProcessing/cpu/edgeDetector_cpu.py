"""
    edge Detection CPU implementation : SOBEL ALGORITHM
    @philipchicco 
"""
# imports
from tools.Picture import Picture
from tools.edgeDetector_Impl import EdgeDetector
import numpy as np
import math


class edgeDetector_cpu(EdgeDetector):

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