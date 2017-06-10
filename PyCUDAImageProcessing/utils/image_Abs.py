"""
image_Abs.py

This module defines the abstract structure of a picture( image) class

"""
import abc


class _Picture(object):

    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def save(self, image, filename=None):
        """
        :param image: image object 
        :param filename: filename
        :return: None
        """

    @abc.abstractmethod
    def width(self):
        """
        :return: image width 
        """

    @abc.abstractmethod
    def height(self):
        """
        :return: image height 
        """

    @abc.abstractmethod
    def get(self, x, y, channel=1):
        """
        return color intensity value at location x,y
        :param x: x-axis location
        :param y: y-axis location
        :return: value
        """

    @abc.abstractmethod
    def set(self, x, y, i, channel=1):
        """
        set color value at co-ordinates
        :param x: x-axis
        :param y: y-axis
        :param i: image object 
        :return: None
        """

    @abc.abstractmethod
    def show(self, i=None, c=None):
        """
        display image in given channel value
        :param i: image object 
        :param c: channel flag
        :return: None
        """

    @abc.abstractmethod
    def load(self, filename, c=None):
        """
        load an image and return an array
        :param filename: 
        :param c: channel flag
        :return: image array
        """

    @abc.abstractmethod
    def get_image_array(self):
        """
        :return: numpy array 
        """

    @abc.abstractmethod
    def validate(self, image):
        """
        Check is file was loaded correctly
        :param image: 
        :return: 
        """

    @abc.abstractmethod
    def intensity(self, x, y):
        """
        calculate intensity of image
        :return: rgb value
        """


