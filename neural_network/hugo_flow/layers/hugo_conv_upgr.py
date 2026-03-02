
import numpy as np
from numpy import ndarray
from numpy.lib.stride_tricks import sliding_window_view
# from ..utils.hugo_utility import Utility as U 

class Conv2d():
    def __init__(self):
        pass

    def Pad_inp(self, inp: ndarray, filter_shape: tuple):

        if len(inp.shape) != 4:
            raise ValueError("Input must be a 4D array (batch_size, channels, height, width)")
        
        if len(filter_shape) != 2:
            raise ValueError("Filter shape must be a tuple of (filter_height, filter_width)")
        
        batch_size, in_channels, in_height, in_width = inp.shape
        to_pad_height = filter_shape[0] - 1
        to_pad_width = filter_shape[1] - 1

        pad_h_top = to_pad_height // 2
        pad_h_bottom = to_pad_height - pad_h_top

        pad_w_left = to_pad_width // 2
        pad_w_right = to_pad_width - pad_w_left

        output = np.pad(inp, 
               ((0, 0), 
                (0, 0), 
                (pad_h_top, pad_h_bottom), 
                (pad_w_left, pad_w_right)), 
               mode='constant', constant_values=0)

        print(output.shape)
        

xd = Conv2d()
xd.Pad_inp(np.random.rand(2, 3, 32, 32), (3, 3))