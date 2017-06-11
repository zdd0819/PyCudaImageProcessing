from utils.image_Abs import _Picture  # Picture Abstract class
import cv2  # opencv3 import
import numpy as np  # array operations
import sys  # system
import os  # os
import string

"""
    Implementation of abstract class image
"""


class Picture(_Picture):
    # numpy array
    image  = None
    width  = None
    height = None
    RED    = 0.299 # red weight average
    GREEN  = 0.587 # green '' ''''
    BLUE   = 0.114 # blue """"""

    def __init__(self, filename=None, width=None, height=None, channel=1):
        try:
            # load image : if error occurs None is returned
            if filename is not None: # no filename provided
                self.filename = filename
                if channel == 3:  # color image
                    self.image = self.load(filename, channel)
                else:  # grayscale
                    self.image = self.load(filename)
            elif width is not None and height is not None: # width and height given
                # create new image
                if width > 0 and height > 0:
                    if channel == 3: # three channels RGB
                        self.image = np.zeros((width, height, 3))
                    else: # grayscale image
                        self.image = np.zeros((width, height))
                else: # error occured
                    print("Invalid height and width NOT provided!")
                    raise
        except IOError as strerror:
            print("IOError: {}".format(strerror))
        except cv2.error as cv_error:
            print("CVError {}".format(cv_error))
        except:
            print("Unexpected error: ", sys.exc_info()[0])
            raise

    def save(self, image, filename=None):
        """
        :param image: image object 
        :param filename: filename
        :return: None
        """
        try:
            if filename is None:  # no name is given
                cv2.imwrite("image.jpg", image)
                print("image saved as : image.jpg")
            else:
                cv2.imwrite(filename, image)
                print("image saved as : ", filename)
        except IOError as strerror:
            print("IOError: {}".format(strerror))
        except:
            print("Unexpected error: ", sys.exc_info()[0])
            raise

    def width(self):
        """
        :return: image width 
        """
        if self.image is not None:
            return np.shape(self.image)[0]
        else:
            return None

    def height(self):
        """
        :return: image height 
        """
        if self.image is not None:
            return np.shape(self.image)[1]
        else:
            return None

    def get(self, x, y, channel=1):
        """
        return color intensity value at location x,y
        :param channel: 
        :param x: x-axis location
        :param y: y-axis location
        :return: value
        """
        if channel == 1:  # grayscale
            if self.image is not None:
                return self.image[x][y]
            else:
                return None
        else:  # color
            if self.image is not None:
                return self.image[x][y][0], self.image[x][y][1], self.image[x][y][2]
            else:
                return None

    def set(self, x, y, i, channel=1):
        """
        set color value at co-ordinates
        :param channel: 
        :param x: x-axis
        :param y: y-axis
        :param i: image object 
        :return: None
        """
        if channel == 1:  # grayscale
            if self.image is not None:
                self.image[x][y] = i
            else:
                print("Error: setting values {0} at {1},{2}".format(x, y, i))
                return
        else:  # color : calculate the intensity value
            if self.image is not None:
                if i is not None:
                    self.image[x][y][0] = i
                    self.image[x][y][1] = i
                    self.image[x][y][2] = i
                else:
                    rgb_value = np.uint8(self.intensity(x, y))
                    self.image[x][y][0] = rgb_value
                    self.image[x][y][1] = rgb_value
                    self.image[x][y][2] = rgb_value
            else:
                print("Error: setting values {0} at {1},{2}".format(x, y, i))
                return

    def show(self, i=None, c=None):
        """
        display image in given channel value
        :param i: image object (must be numpy array)
        :param c: window name
        :return: None
        """
        try:
            if i is None and self.image is not None:  # no image object: display class object
                if c is None:  # display : no window name
                    cv2.imshow("image", self.image, )
                else:
                    assert isinstance(self.filename, string)
                    c = self.filename
                    cv2.imshow(c, self.image)
            elif i is None and self.image is None:
                print("No image object loaded : please provide an image")
                return
            else:
                if c is None:  # display : no window name
                    cv2.imshow("image", i)
                else:
                    cv2.imshow(c, i)
            cv2.waitKey(0)
        except IOError as strerror:
            print("IOError: {}".format(strerror))
        except:
            print("Unexpected error: ", sys.exc_info()[0])
            raise

    def load(self, filename, c=None):
        """
        load an image and return an array
        :param filename: 
        :param c: channel flag
        :return: image array
        """
        try:
            if c is None: # gray
                img = self.validate(cv2.imread(filename, 0))
                return self.resize_image(img)
            else: # color
                img = self.validate(cv2.imread(filename, 1))
                return self.resize_image(img)
        except IOError as strerror:
            print("IOError: {}".format(strerror))
        except:
            print("Unexpected error: ", sys.exc_info()[0])
            raise

    def get_image_array(self):
        """
        :return: numpy array 
        """
        if self.image is not None:
            return self.image
        else:
            return None

    def validate(self, image):
        """
        Check is file was loaded correctly
        :param image: 
        :return: 
        """
        if image is None:
            print("Invalid filename/ path : ensure path is correct")
            return
        else:
            return image

    def intensity(self, x, y):
        """
        :param image: image array object
        :param x: x-co-ordinates 
        :param y: y-co-ordinates
        :return: new rgb intensity value
        """
        # check for None and validate RGB channels
        if self.image is not None:
            if len(self.image.shape) > 0:  # not gray
                R, G, B = self.get(x=x, y=y, channel=3) # get values
                return R * self.RED + G * self.GREEN + B * self.BLUE
            else:
                print("Error in intensity calculation: invalid co-ordinates and channel")
                return
        else:
            print("Error: Object is not member of numpy.ndarray , is empty")
            return

    def rgb(self, r, g, b):
        return r * self.RED + g * self.GREEN + b * self.BLUE

    def resize_image(self, i):
        """
        resize image to standard size
        :param i: image object
        :return: rezised image
        """
        if i is None:
            print("Image is None TYPE. ERROR")
            return
        else:
            w, h = i.shape
            if not (h % 32) == 0:
                h_new = int(h / 32)
                if h_new > 16:  # greater than 1024
                    h = 16 * 32
                else:
                    h = h_new * 32
            elif int(h / 32) > 16:
                h = 16 * 32

            if not (w % 32) == 0:
                w_new = int(w / 32)
                if w_new > 16:  # greater than 1024
                    w = 16 * 32
                else:
                    w = w_new * 32
            elif int(w / 32) > 16:
                w = 16 * 32

        return cv2.resize(i, (w, h), interpolation=cv2.INTER_AREA)
