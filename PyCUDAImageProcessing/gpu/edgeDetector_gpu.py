"""
    edge Detection GPU implementation : SOBEL ALGORITHM
    @philipchicco 
"""
# cpu host imports
from tools.Picture import Picture
from tools.edgeDetector_Impl import EdgeDetector
import numpy as np
import math
import string

# gpu device imports : NVIDIA CUDA
import pycuda.autoinit  # memory management
import pycuda.driver as drv  # cuda driver
import pycuda.gpuarray as gpuarray  # gpu array handle
from pycuda.compiler import SourceModule  # Module wrapper for C


class edgeDetector_gpu(EdgeDetector):
    def __init__(self, kernel_code=None):
        self.kernel_code = kernel_code
        # to be updated

    def device_kernel(self, width, height, kernel_code=None):
        """
        Define device kernel function and compile
        :param height: image height
        :param width: image width
        :param kernel_code: kernel function definition
        :return: kernel code handle
        """
        if kernel_code is None:  # use default
            self.kernel_code = """
            // add code here
            """
        else:
            assert isinstance(kernel_code, str)
            self.kernel_code = kernel_code

        # get kernel_code
        # compile code
        template_code = self.kernel_code % {
            'WIDTH': width,
            'HEIGHT': height
        }
        #print(self.kernel_code)
        self.module = SourceModule(template_code)
        return self.module.get_function("sobel_edges")

    def sobel_edges(self, image, channel=1):
        """
        GPU IMplementation of sobel algorithm
        :param image: image ndarray
        :param channel: color channels
        :return: edged image array
        """

        # allocate gpu device memory for array: convert to float32
        # auto memory allocation
        if isinstance(image, np.ndarray):  # its a must
            lw, lh = image.shape
            array_gpu_In = gpuarray.to_gpu(image.flatten().astype(np.float32)).astype(np.float32)
            array_gpu_Out = gpuarray.empty((lw, lh), dtype=np.float32).astype(np.float32)
        else:
            print("Error: image array is not instance of numpy.ndarray")
            return

        # configurations
        # block
        bdim = (32, 32, 1)  # x, y, z
        dx, mx = divmod(lh, bdim[0])  # cols
        dy, my = divmod(lw, bdim[1])  # rows

        # grid
        gdim = ((dx + (mx > 0)) * bdim[0], (dy + (my > 0)) + bdim[1])

        # execute kernel
        kernel = self.device_kernel(lw, lh, kernel_code=self.sobel_edges_kernel())  # compile
        # calculate time
        start = drv.Event()
        end = drv.Event()
        start.record()
        # kernel execution
        kernel(array_gpu_In, array_gpu_Out, block=bdim, grid=gdim)
        end.record()
        end.synchronize()
        start.synchronize()
        # collect result : convert back to original format
        return (array_gpu_Out.get().reshape(lw, lh)).astype(np.float32), start.time_till(end) + 1e-3

    def brightness(self):
        kernel_code = """
        #include <math.h>
        
        __global__ void brightness(float *matrix_in, float *matrix_out)
        {   
            // 2D threads
            int pos_y = blockIdx.y * blockIdx.y + threadIdx.y;
            int pos_x = blockIdx.x * blockIdx.x + threadIdx.x;
            
            // brightness value = alpha * pixel + beta
            if (pos_y < %(HEIGHT)s && pos_x < %(WIDTH)s){
                float value = (2.0 * matrix_in[pos_y * %(WIDTH)s + pos_x]) + 100.0;    
                matrix_out[pos_y * %(WIDTH)s + pos_x] = value;
            }

        }
        """
        return kernel_code

    def sobel_edges_kernel(self):
        kernel = """
        #include <math.h>
    
        __global__ void sobel_edges(float *matrix_in, float *matrix_out)
        {
            
            // pixel location
            int pos_y = blockIdx.y * blockDim.y + threadIdx.y;
            int pos_x = blockIdx.x * blockDim.x + threadIdx.x;
            int ret_x = 0;
            int ret_y = 0;
            int res   = 0;
                
            //
            if ( (pos_y >= 0 && pos_y < %(HEIGHT)s) && (pos_x >= 0 && pos_x < %(WIDTH)s) )
            {
                ret_x += -matrix_in[%(WIDTH)s *(pos_y - 1) + (pos_x - 1)] + matrix_in[%(WIDTH)s * (pos_y -1)+(pos_x+1)] 
                         -2*matrix_in[%(WIDTH)s *(pos_y) + (pos_x - 1)] + 2*matrix_in[%(WIDTH)s * (pos_y)+(pos_x+1)]
			             -matrix_in[%(WIDTH)s *(pos_y + 1) + (pos_x - 1)] + matrix_in[%(WIDTH)s * (pos_y +1)+(pos_x+1)];
                
                ret_y += matrix_in[%(WIDTH)s * (pos_y-1) + (pos_x-1)] + 2*matrix_in[%(WIDTH)s *(pos_y-1)+(pos_x+1)] +
                         matrix_in[%(WIDTH)s * (pos_y-1)+(pos_x+1)] - matrix_in[%(WIDTH)s *(pos_y+1)+(pos_x-1)] 
                         -2*matrix_in[%(WIDTH)s * (pos_y+1)+(pos_x)] - matrix_in[%(WIDTH)s *(pos_y+1)+(pos_x+1)];
                
                ret_x = ret_x/5;
                ret_y = ret_y/5;
                 
                res = (int)sqrtf(powf((float)ret_x, 2) + powf((float)ret_y, 2));
                
                if (res > 255)
                    res = 255;
                
                matrix_out[pos_y * %(WIDTH)s + pos_x] = res;
            }
        }
        // end of definitions
        """
        return kernel
