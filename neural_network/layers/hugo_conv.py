
import numpy as np
from numpy import ndarray
from numpy.lib.stride_tricks import sliding_window_view
from utils.hugo_utility import Utility as U 
import time


class Conv_layer():
        def __init__(self, model = None):
           self.model = model
           self.layer_set = False
           self.activation_functions = {'none': U.no_activation_function, 'sigmoid' : U.sigmoid, 'relu' : U.relu, 'leaky relu': U.leaky_relu, 'tanh': U.tanh}
           self.weights_initializations = {'linear' : U.linear, 'he' : U.he, 'xavier': U.xavier}
           self.clipping_methods = {'norm clipping': U.norm_clipping}
           self.update_methods = {'gradient descent': U.basic_grad_update, 'SGD': U.SGD_momentum}
           
           
           

        def set_layer(self, param, activation_function = 'none', weight_initialization = None, update_method = 'gradient descent', jump: int = 0, filters: int = 1, 
                      input_layer = False, sequential = True):
            
            if isinstance(param, np.ndarray):
                self.params = [param for _ in range(filters)]
            else:
              self.params = [np.random.uniform(-1, 1, param) for _ in range(filters)]

            if update_method == 'gradient descent' and self.model.update_method !='gradient descent':
                self.update_method = self.model.update_method
            else:
                self.update_method = update_method

            self.bias = 0
            self.update_method = update_method
            self.jump = jump
            self.activation_function = activation_function
            self.layer_af_calc = self.activation_functions[activation_function]
            self.weight_initialization = weight_initialization
            self.filters = filters
            self.layer_set = True
            self.velocity_w = 0
            self.velocity_b = 0
            self.input_l = input_layer
            self.sequential = sequential
                 
           
        def forward_L(self, input, training):
           
           if input.ndim != 4:
              print(f'Input to conv_layer must be shape (Samples, channels, widht, height).\n Not {input.shape} ')
              raise LookupError()

           if self.layer_set != True:
              print('Conv_layer is not set. Use "Conv_layer.set_layer" before anything else!')
              raise LookupError()
    
           self.inp_shape = input.shape
           self.input_grads = np.zeros(input.shape)
           self.weight_grads = []
          
               
           output_sample, input_pad, self.patches = conv_ld(inp = input, params =self.params, bias = self.bias, jump = self.jump, 
                                                            filters_amount= self.filters, sequential= self.sequential)
           
        

           self.flatten = np.reshape(output_sample, (output_sample.shape[0], -1))
           self.flatten, self.af_gradient = self.layer_af_calc(self.flatten) 
           
           print(self.flatten.shape)
        #    quit()
           return self.flatten


        def backward_L(self, grad):
        
           grad = grad * self.af_gradient     

           if self.sequential == True:
             for patch, param in zip(reversed(self.patches), reversed(self.params)):

               layer_weight_grad = np.dot(grad.flatten(), patch).reshape(self.params[0].shape)
               self.velocity_w = self.update_methods[self.update_method](self.model.lr,  layer_weight_grad,  self.velocity_w)
               param -= layer_weight_grad * self.model.lr

               param_flipped = np.flip(param)
               grad, no_matter, no_matter_2 = conv_ld(inp = np.reshape(grad, self.inp_shape), params = param_flipped, bias = 0, single_param = True)

           else:
               
               
        
               patches = np.stack(self.patches).transpose(0, 2, 1)   
               params = np.stack(self.params)
               grad_upd = np.reshape(grad, (grad.shape[0], self.inp_shape[2], self.inp_shape[3], self.filters))
               grad_upd = grad_upd.transpose(3, 0, 1, 2)
               grad_upd = grad_upd.reshape(self.filters, -1)  
               
               
               
               layer_weight_grad = np.einsum('ijk,ik->ij', patches, grad_upd).reshape(self.filters, *self.params[0].shape)
               for i, param in enumerate(self.params):
                  param -= layer_weight_grad[i] * self.model.lr

               if not self.input_l:
                  grad_summed = []
                  for param in self.params:
                    param_flipped = np.flip(param)
                    grad_to_sum, no_matter, no_matter_2 = conv_ld(inp = np.reshape(grad, self.inp_shape), params = param_flipped, bias = 0, single_param = True)
                    grad_summed.append(grad_to_sum)
                  grad = np.sum(grad_summed,axis=0)

           layer_bias_grad = np.sum(grad) 
           self.velocity_b = self.update_methods[self.update_method](self.model.lr,  layer_bias_grad,    self.velocity_b)
           self.bias -= layer_bias_grad * self.model.lr
           
           return grad

           
        
           
        
        def get_info(self):
            print(f'self.kernels{self.kernels.shape}')
            print(f'self.kernels_weights{self.kernels_weights.shape}')



def conv_ld(inp: ndarray, params: ndarray, bias: float, filters_amount: int = 1, 
            jump: int = 0, single_param = False, sequential = True) -> ndarray:
    """
    Custom convolution-like function.

    Args:
        inp: Input tensor (expected shape: (samples, channels, height, width)).
        params: Convolution filters.
        bias: Bias term to add after convolution.
        filters_amount: Number of filters to apply (used in parallel mode).
        jump: Currently unused.
        single_param: If True, apply only one filter across input.
        sequential: If True, apply filters one after another (like a conv block).

    Returns:
        output: Convolution result.
        input_pad: Padded input used for the last convolution.
        patches_stack: List of flattened patch matrices from sliding windows.
    """

    # ---------------------------------------------
    # CASE 1: SINGLE FILTER USED ACROSS INPUT
    # ---------------------------------------------
    if single_param == True:
        # Pad the input based on filter size
        input_pad = input_pad_calc(inp, params)

        # Create sliding windows: shape becomes (samples, channels, new_h, new_w, f_h, f_w)
        patches = sliding_window_view(input_pad, (params.shape[0], params.shape[1]), axis=(1, 2))

        # Flatten patches to shape: (num_patches, filter_size)
        patches = np.reshape(patches, (patches.size // params.size, params.size))

        # Apply dot product between all patches and filter weights
        output = np.dot(patches, params.flatten())

        # Add bias
        output = output + bias

        # Reshape output to match input's spatial dimensions
        output = np.reshape(output, inp.shape)

        return output, input_pad, patches

    # ---------------------------------------------
    # CASE 2: SEQUENTIAL MODE (like a stack of convs)
    # Each filter output becomes the input to the next
    # ---------------------------------------------
    if sequential == True:
        patches_stack = []

        # Iterate over filters
        for param in params:
            # Pad input to preserve dimensions
            input_pad = input_pad_calc(inp, param)

            # Extract sliding windows across input
            patches = sliding_window_view(input_pad, (param.shape[0], param.shape[1]), axis=(1, 2))

            # Flatten patches
            patches = np.reshape(patches, (patches.size // param.size, param.size))

            # Store patches
            patches_stack.append(patches)

            # Apply filter
            output = np.dot(patches, param.flatten())

            # Add bias
            output = output + bias

            # Reshape output to match input (assuming same shape)
            output = np.reshape(output, inp.shape)

            # Output becomes input for next filter (sequential stacking)
            inp = output

    # ---------------------------------------------
    # CASE 3: PARALLEL FILTERING (multi-channel conv)
    # Each filter is applied independently, result stacked
    # ---------------------------------------------
    else:
        patches_stack = []

        # Initialize output tensor for all filters
        # Shape: (filters, samples, height, width)
        output = np.zeros((filters_amount, inp.shape[0], inp.shape[2], inp.shape[3]))

        for idx, param in enumerate(params):
            # Pad input per filter
            input_pad = input_pad_calc(inp, param)

            # Get patches from input. Patches shape: (samples, height, width, filter_h, filter_w)
            patches = sliding_window_view(input_pad, param.shape, axis=(1, 2))
            
            # Flatten patches to shape: (patches_amount, param.size)
            patches = np.reshape(patches, (patches.size // param.size, param.size))
            
            # Store patches
            patches_stack.append(patches)

            # Apply filter
            output_stack = np.dot(patches, param.flatten())
            output_stack = output_stack + bias

            # Normalize if needed (commented out)
            # output_stack = output_stack / np.max(np.abs(output_stack))

            # Reshape output to (samples, height, width)
            output_stack = np.reshape(output_stack, (inp.shape[0], inp.shape[2], inp.shape[3]))

            # Save into output array
            output[idx] = output_stack
        output = output.transpose(1,0,2,3)
    # Output shape for parallel mode:
    # (filters, samples, height, width)
    
    return output, input_pad, patches_stack


def _pad_ld(inp: ndarray, param_size: int) -> ndarray:

    z = np.array([0])
    
    z_to_add = inp.size - (inp.size - (param_size - 1))
    for i in range(z_to_add):
        
        if i % 2 == 1:
         inp = np.concatenate([z, inp])
        if i % 2 == 0:
         inp = np.concatenate([inp, z])

    return inp


# inp = input param = filter
def input_pad_calc(inp: ndarray, param: ndarray, jump: int = 0) -> ndarray:
     # filling entry data
    
    
    param_len_0 = param.shape[0]
    param_len_1 = param.shape[1]

    channels_combined_list = []
    # print(inp[0].shape)
    # quit()
    for inx, channel in enumerate(inp[0]):

        channel_pad_list = []
        
        for column in channel.T:
            channel_pad = _pad_ld(column, param_len_0)
            channel_pad_list.append(channel_pad)

        channel_pad_real = np.array(channel_pad_list).T
        channel_pad_list = []

        for row in channel_pad_real:
            channel_pad = _pad_ld(row, param_len_1)
            channel_pad_list.append(channel_pad)
        
       
        
        channel_pad = np.array(channel_pad_list)
        channels_combined_list.append(channel_pad) 
    
    
    channels_combined = np.stack(channels_combined_list)
    # print(channels_combined.shape)
    
    # quit()
    samples = inp
    
    padded_batch = U.channels_pad_batch(samples = samples, example = channels_combined[0])
    # print(inp.shape)
    # print(padded_batch.shape)
    
    padded_batch = np.reshape(padded_batch, (inp.shape[0], padded_batch.shape[1], padded_batch.shape[2]))
    # print(padded_batch.shape)
    # quit()
    return padded_batch

   


         



    
    
    








 




































































































