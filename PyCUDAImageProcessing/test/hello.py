import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule
import cv2


# mod = SourceModule("""
# __global__ void multiply_them(float *dest, float *a, float *b)
# {
#   const int i = threadIdx.x;
#   dest[i] = a[i] * b[i];
# }
# """)
#
# multiply_them = mod.get_function("multiply_them")
#
# a = numpy.random.randn(400).astype(numpy.float32)
# b = numpy.random.randn(400).astype(numpy.float32)
#
# dest = numpy.zeros_like(a)
# multiply_them(
#         drv.Out(dest), drv.In(a), drv.In(b),
#         block=(400, 1, 1), grid=(1,1))
#
# print(dest-a*b)

# read image into matrix.
#m = cv2.imread("1.jpg")

# get image properties.
#w, h, bpp = np.shape(m)

# print image properties.
#print("width: " + str(w))
#print("height: " + str(h))
#print("bpp: " + str(bpp))
#print(np.shape(m)[0])

from tools.Picture import Picture

from scipy import misc
import math

x = np.array([
                [1, 2, 3],[4, 5, 6],[7, 8, 9]
            ]).flatten()
print(x)
for i in range(-1,2):
	for j in range(-1,2):
		print(x[((i+1) * 3) + (j+1)])
