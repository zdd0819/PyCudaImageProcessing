"""
    Run the entire program: programmtically pass values or 
    use argument parser or design a UI. (entirely up to you)
    
    Usage: python main.py image.jpg 
    
"""

# imports
from cpu.edgeDetector_cpu import edgeDetector_cpu
from gpu.edgeDetector_gpu import edgeDetector_gpu
from tools.Picture import Picture
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import misc
import time


# arguments
IMAGE_TEST = "images\\1.jpg"

def main():

    # detector handle : loaded with default filters
    detector = edgeDetector_cpu()
    det_GPU = edgeDetector_gpu()
    print("LOG: using CPU detection --- ")
    picture_1 = Picture(IMAGE_TEST, channel=1)
    #print(isinstance("dsdsds", str))
    new_image = det_GPU.sobel_edges(picture_1.get_image_array())

    """
    print("LOG: image loaded --- ")
    print("LOG: image dimensions - ", picture_1.width(), " x ", picture_1.height())

    # measure time
    start_time = time.time()
    new_image = detector.sobel_edges(picture_1, channel=1)
    elapsed_time = time.time() - start_time
    print("LOG : Elapsed_time_CPU : " + str(elapsed_time))

    # save the image
    #misc.imsave(name="results\\_edged_" + IMAGE_TEST.split('\\')[-1], arr=new_image.get_image_array())
    """
    #misc.imsave(name="results\\_edged_GPU" + IMAGE_TEST.split('\\')[-1], arr=new_image)

    # display
    print("LOG : Displaying results: ---------- ")
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax1.imshow(picture_1.get_image_array().astype(np.float32), cmap=plt.cm.gray, vmin=30, vmax=200)
    ax2 = fig.add_subplot(122)
    ax2.imshow(new_image, cmap=plt.cm.gray, vmin=30, vmax=200)
    plt.show()



    print("LOG: terminated successfully")

# runner
if __name__ == "__main__":
    main()