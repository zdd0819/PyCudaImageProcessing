"""
Abstract class for Edge Detector
"""
from abc import ABCMeta, abstractmethod


class _EdgeDetector(object):

    __metaclass__ = ABCMeta


    @abstractmethod
    def truncate(self, value):
        """
        :param value: a pixel value 
        :return: a truncated value of the pixel
        """

    @abstractmethod
    def set_filter_1(self, array):
        """
        set the filter 1 values
        :param array: 
        :return: None 
        """

    @abstractmethod
    def set_filter_2(self, array):
        """
        set the filter 2 values
        :param array: 
        :return: None
        """

    @abstractmethod
    def get_filter_1(self):
        """
        filter 1 array
        :return: numpy array
        """

    @abstractmethod
    def get_filter_2(self):
        """
        filter 2 array
        :return: numpy array
        """

    @abstractmethod
    def calculateEdges(self, image):
        """
        calculate the edges in a given image
        :param image: 
        :return: new image with edge detection 
        """