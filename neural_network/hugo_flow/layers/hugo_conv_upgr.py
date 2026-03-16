
import numpy as np
from numpy import ndarray
from numpy.lib.stride_tricks import sliding_window_view
from ..utils.hugo_utility import Utility as U 

class Conv2d():
    def __init__(self):
        
        self.activation_functions = {'linear': U.no_activation_function, 'sigmoid' : U.sigmoid, 'relu' : U.relu, 'leaky relu': U.leaky_relu, 'tanh': U.tanh}
        self.loss_methods = {'mse': U.mse_loss, 'cross_entropy': U.cross_entropy_loss}
        self.update_methods = {'gradient descent': U.basic_grad_update, 'SGD': U.SGD_momentum}
        # self.loss = loss
        # self.lr = lr
        self.weights_initializations = {'linear' : U.linear, 'he' : U.he, 'xavier': U.xavier}
      
        pass

    def set_layer(self, filter_shape = (3,3), filters_amount = 1, weight_initialization = None, filters_shapes_list = []):
        self.filter_shape = filter_shape
        if len(self.filter_shape) != 2:
            raise ValueError("Filter shape must be a tuple of (filter_height, filter_width)")
        
        self.filters_amount = filters_amount
        self.weight_initialization = weight_initialization


        self.weights = np.random.randn(self.filters_amount, *filter_shape) * self.weights_initializations[weight_initialization](filter_shape[0], filter_shape[1])
        
        self.bias = np.zeros(self.filters_amount,)


    def Pad_inp(self, inp: ndarray, ):

        if len(inp.shape) != 4:
            raise ValueError("Input must be a 4D array (batch_size, channels, height, width)")
        
        
        

        k_h, k_w = self.filter_shape
        
        batch_size, in_channels, in_height, in_width = inp.shape
        to_pad_height = k_h - 1
        to_pad_width = k_w - 1

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

        return output
    
    def forward_L(self, inp: ndarray):

        if len(inp.shape) != 4:
            raise ValueError("Input must be a 4D array (batch_size, channels, height, width)")
        
        batch_size, in_channels, in_height, in_width = inp.shape
        
        if not hasattr(self, 'weights'):
          self.weights = np.random.randn(self.filters_amount, in_channels, *self.filter_shape) * self.weights_initializations[self.weight_initialization](self.filter_shape[0], self.filter_shape[1])
        if not hasattr(self, 'bias'):
          self.bias = np.zeros(self.filters_amount,)

        padded_inp = self.Pad_inp(inp)
        windows =  sliding_window_view(padded_inp, self.filter_shape, axis=(2, 3))
        # Nakłdamy filtry (wagi)
        conv_out = np.tensordot(windows, self.weights, axes=((1, 4, 5), (1, 2, 3)))



xd = Conv2d()
xd.Pad_inp(np.random.rand(2, 3, 32, 32), (3, 3))