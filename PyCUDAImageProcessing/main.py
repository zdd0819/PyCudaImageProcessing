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
SAVE = True
MODE = "CPU"
DISPLAY = False

def main(MODE="CPU"):

    try:
        det_GPU = None
        det_CPU = None

        # detector handle : loaded with default filters
        if MODE == "CPU":
            det_CPU = edgeDetector_cpu()
        else:
            det_GPU = edgeDetector_gpu()

        # load image
        pic = Picture(IMAGE_TEST, channel=1)
        if pic is not None:
            print("LOG: Loaded image sucessfully")
            print("LOG: image dimensions - ", pic.width(), " x ", pic.height())
            if det_CPU is not None:
                #
                MODE = "CPU"
                print("LOG: using CPU detection --- ")
                # measure time
                start_time = time.time()
                new_image = det_CPU.sobel_edges(pic, channel=1)
                elapsed_time = time.time() - start_time
                print("LOG : Elapsed_time_CPU : " + str(elapsed_time))
                if SAVE:
                    # save the image
                    misc.imsave(name="results\\_edged_CPU_" + IMAGE_TEST.split('\\')[-1],
                                arr=new_image)
            elif det_GPU is not None:
                MODE = "GPU"
                print("LOG: using GPU detection --- ")
                new_image, time_gpu = det_GPU.sobel_edges(image=pic.get_image_array(), channel=1)
                print("LOG : Elapsed_time_GPU : " + str(time_gpu))
                if SAVE:
                    # save the image
                    misc.imsave(name="results\\_edged_GPU_" + IMAGE_TEST.split('\\')[-1],
                                arr=new_image)
        # display
            fig = plt.figure()
            ax1 = fig.add_subplot(121)
            fig.suptitle("Edge Detection"+ MODE)
            ax1.imshow(pic.get_image_array().astype(np.float32), cmap=plt.cm.gray, vmin=30, vmax=200)
            ax2 = fig.add_subplot(122)
            ax2.imshow(new_image, cmap=plt.cm.gray, vmin=30, vmax=200)
            if MODE == "CPU" and SAVE:
                fig.savefig("results\\fig_edged_CPU_" + (IMAGE_TEST.split('\\')[-1]).split(".")[0] + ".png")
            elif MODE == "GPU" and SAVE:
                fig.savefig("results\\fig_edged_GPU_" + (IMAGE_TEST.split('\\')[-1]).split(".")[0] + ".png")
            if DISPLAY:
                print("LOG : Displaying results: ---------- ")
                plt.show()

        print("LOG: ----------------  terminated successfully ---------------------- ")
    except Exception as e:
        print("Exception occured ", e)
        return

# runner
if __name__ == "__main__":
    main("GPU")