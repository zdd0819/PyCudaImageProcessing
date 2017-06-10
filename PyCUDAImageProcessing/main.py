"""
    Run the entire program: programmtically pass values or 
    use argument parser or design a UI. (entirely up to you)
    
    Usage: python main.py image.jpg 
    
"""

# imports
from cpu.edgeDetector_cpu import edgeDetector_cpu
from tools.Picture import Picture
import numpy as np


# arguments
IMAGE_TEST = "images\\lena3.jpg"


def main():

    # detector handle : loaded with default filters
    detector = edgeDetector_cpu()
    print("-- detector passed")

    # picture handle
    picture = Picture(IMAGE_TEST)


    picture_1 = Picture(IMAGE_TEST, channel=1)

    #print(picture_1.load(IMAGE_TEST, "COLOR").shape)
    #print(picture_1.get(25, 96, channel=3))
    #picture_1.get_image_array()[12][12][0] = 14

    print("-- picture passed")

    # compute edges and display
    print(detector.calculateEdges(picture_1, channel=1).show())










# runner
if __name__ == "__main__":
    main()