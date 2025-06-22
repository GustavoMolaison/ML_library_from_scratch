
import numpy as np
from numpy import ndarray
from numpy.lib.stride_tricks import sliding_window_view
from utils.hugo_utility import Utility as U 
import time

class Conv_layer():
    def __init__(self, model=None):
        self.model = model
        self.layer_set = False

        # Activation functions dictionary
        self.activation_functions = {
            'none': U.no_activation_function,
            'sigmoid': U.sigmoid,
            'relu': U.relu,
            'leaky relu': U.leaky_relu,
            'tanh': U.tanh
        }
        # Weight initialization methods dictionary
        self.weights_initializations = {
            'linear': U.linear,
            'he': U.he,
            'xavier': U.xavier
        }
        # Clipping methods dictionary
        self.clipping_methods = {'norm clipping': U.norm_clipping}

        # Update methods dictionary (e.g., SGD, momentum)
        self.update_methods = {
            'gradient descent': U.basic_grad_update,
            'SGD': U.SGD_momentum
        }

    def set_layer(self, param, activation_function='none', weight_initialization=None,
                  update_method='gradient descent', jump: int = 0, filters: int = 1,
                  input_layer=False, sequential=True, flat_output=True, bias=0):
        """
        Setup the layer parameters, activation, update method, and filters.
        """
        # Initialize filters: copy param if ndarray else random init
        if isinstance(param, np.ndarray):
            self.params = [np.copy(param) for _ in range(filters)]
        else:
            self.params = [np.random.uniform(-1, 1, param) for _ in range(filters)]

        # Decide which update method to use, prefer model's if different from default
        if update_method == 'gradient descent' and self.model.update_method != 'gradient descent':
            self.update_method = self.model.update_method
        else:
            self.update_method = update_method

        self.bias = bias
        self.jump = jump
        self.activation_function = activation_function
        self.layer_af_calc = self.activation_functions[activation_function]
        self.weight_initialization = weight_initialization
        self.filters = filters
        self.layer_set = True
        self.velocity_w = 0  # Velocity for weight updates (momentum)
        self.velocity_b = 0  # Velocity for bias updates
        self.input_l = input_layer
        self.sequential = sequential
        self.flat_output = flat_output

    def forward_L(self, input, training):
        """
        Forward pass through convolutional layer.
        Args:
            input: Input tensor of shape (samples, channels, height, width)
            training: Boolean flag if training or inference mode
        Returns:
            output: activated output, flattened if flat_output=True
        """
        if input.ndim != 4:
            raise ValueError(f'Input to conv_layer must be shape (Samples, channels, width, height).\n Not {input.shape} ')

        if self.layer_set != True:
            raise ValueError('Conv_layer is not set. Use "Conv_layer.set_layer" before anything else!')

        self.inp_shape = input.shape

        # Initialize biases if not already done
        if not hasattr(self, 'biases'):
            self.biases = np.zeros(self.filters)

        # Perform convolution-like operation
        output_sample, self.patches = conv_ld(
            inp=input,
            params=self.params,
            bias=self.biases,
            jump=self.jump,
            filters_amount=self.filters,
            sequential=self.sequential
        )

        # Flatten output spatial dimensions
        output_flatten = np.reshape(output_sample, (output_sample.shape[0], -1))
        # Apply activation function and get gradient for backprop
        output_flatten, self.af_gradient = self.layer_af_calc(output_flatten)

        if self.flat_output:
            return output_flatten
        else:
            return np.reshape(output_flatten, output_sample.shape)

    def backward_L(self, grad):
        """
        Backward pass: compute gradients and update weights and biases.
        Args:
            grad: Gradient from next layer, shape compatible with output.
        Returns:
            input_grad: Gradient w.r.t input for previous layer or 0 if input layer.
        """
        # Element-wise multiply grad with activation function gradient
        grad = np.reshape(np.reshape(grad, (grad.shape[0], -1)) * self.af_gradient, grad.shape)

        if self.sequential:
            # Sequential mode: filters applied one after another
            input_grad = grad
            for idx, (patch, param) in enumerate(zip(reversed(self.patches), reversed(self.params))):
                # Calculate weight gradient via dot product between grad and patch
                layer_weight_grad = np.dot(input_grad.flatten(), patch).reshape(self.params[0].shape)

                # Update weight velocity and apply gradient update
                self.velocity_w = self.update_methods[self.update_method](self.model.lr, layer_weight_grad, self.velocity_w)
                param -= layer_weight_grad * self.model.lr

                # Compute bias gradient and update bias velocity & biases
                layer_bias_grad = np.sum(input_grad)
                self.velocity_b = self.update_methods[self.update_method](self.model.lr, layer_bias_grad, self.velocity_b)
                self.biases[idx] -= layer_bias_grad * self.model.lr

                # Flip filter to compute gradient w.r.t input and apply conv_ld backward step
                param_flipped = np.flip(param)
                input_grad, _ = conv_ld(
                    inp=np.reshape(input_grad, self.inp_shape),
                    params=param_flipped,
                    bias=0,
                    single_param=True,
                    sequential=True
                )

        else:
            # Parallel mode: multiple filters applied simultaneously
            start_time = time.time()

            # Rearrange patches and gradient for einsum multiplication
            patches = self.patches.transpose(0, 2, 1)  # (filters, samples, patch_size)

            grad_upd = np.reshape(grad, (grad.shape[0], self.inp_shape[2], self.inp_shape[3], self.filters))
            grad_upd = grad_upd.transpose(3, 0, 1, 2)  # (filters, batch, height, width)
            grad_upd = grad_upd.reshape(self.filters, -1)  # (filters, num_output_positions)

            # Compute weight gradients for all filters at once using einsum
            layer_weight_grad = np.einsum('ijk,ik->ij', patches, grad_upd).reshape(self.filters, *self.params[0].shape)

            # Update weights for each filter
            for i, param in enumerate(self.params):
                param -= layer_weight_grad[i] * self.model.lr

            # Compute and update bias gradients
            layer_bias_grad = np.sum(grad, axis=0)
            self.velocity_b = self.update_methods[self.update_method](self.model.lr, layer_bias_grad, self.velocity_b)

            layer_bias_grad = np.reshape(layer_bias_grad, (self.filters, self.inp_shape[2] * self.inp_shape[3]))
            for idx, b_grad in enumerate(layer_bias_grad):
                self.biases[idx] -= np.sum(b_grad) * self.model.lr

            # If not input layer, compute gradient w.r.t input for previous layer
            if not self.input_l:
                param_flipped = [np.flip(param) for param in self.params]
                grad_conv = np.reshape(grad, (self.inp_shape[0], -1, self.inp_shape[2], self.inp_shape[3]))
                input_grad, _ = conv_ld(
                    inp=grad_conv,
                    params=param_flipped,
                    bias=np.zeros(self.biases.shape),
                    filters_amount=grad_conv.shape[1],
                    single_param=False,
                    sequential=False
                )

            end_time = time.time()
            print(f"Execution time backprop parallel: {end_time - start_time:.6f} seconds")

        if not self.input_l:
            return input_grad  # Gradient to pass to previous layer
        else:
            return 0  # No input gradient if this is the input layer



def conv_ld(inp: ndarray, params: ndarray, bias: ndarray, filters_amount: int = 1, 
            jump: int = 0, single_param = False, sequential = False, backprop = False) -> ndarray:
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
    # CASE 1: single filter across input
    if single_param == True:
        input_pad = input_pad_calc(inp, params)
        patches = sliding_window_view(input_pad, (params.shape[0], params.shape[1]), axis=(1, 2))
        patches = np.reshape(patches, (patches.size // params.size, params.size))
        output = np.dot(patches, params.flatten())
        output = output + bias
        output = np.reshape(output, (inp.shape[0], inp.shape[2], inp.shape[3]))
        return output, None


    output = np.zeros((filters_amount, inp.shape[0], inp.shape[2], inp.shape[3]))

    # CASE 2: Sequential application of filters
    if sequential == True:
        output = inp
        for idx, param in enumerate(params):
            # Pad input per filter size
            input_pad = input_pad_calc(output, params[0])
            # Extract sliding window patches from input
            patches = sliding_window_view(input_pad, params[0].shape, axis=(1, 2))
            patches = np.reshape(patches, (patches.size // params[0].size, params[0].size))

            output = np.dot(patches, param.flatten())
            output = output + bias[idx]
            output = np.reshape(output, inp.shape)
            

           
            

    # CASE 3: Parallel filtering (independent filters)
    else:
        for idx, param in enumerate(params):
            # Pad input per filter size
            input_pad = input_pad_calc(inp, params[0])
            # Extract sliding window patches from input
            patches = sliding_window_view(input_pad, params[0].shape, axis=(1, 2))
            patches = np.reshape(patches, (patches.size // params[0].size, params[0].size))
            output_stack = np.dot(patches, param.flatten())
            output_stack = output_stack + bias[idx]
            output_stack = np.reshape(output_stack, (inp.shape[0], inp.shape[2], inp.shape[3]))
            output[idx] = output_stack
        output = output.transpose(1, 0, 2, 3)

    # Repeat patches for each filter for potential backprop use
    patches = np.repeat(patches[np.newaxis, :, :], repeats=filters_amount, axis=0)

    return output, patches


def _pad_ld(inp: ndarray, param_size: int) -> ndarray:
    """
    Pad 1D array on both sides to accommodate convolution window size.
    """
    z = np.array([0])
    z_to_add = inp.size - (inp.size - (param_size - 1))
    for i in range(z_to_add):
        if i % 2 == 1:
            inp = np.concatenate([z, inp])
        if i % 2 == 0:
            inp = np.concatenate([inp, z])
    return inp


def input_pad_calc(inp: ndarray, param: ndarray, jump: int = 0) -> ndarray:
    """
    Pad the input tensor spatially according to the filter size to enable full convolution.
    """
    param_len_0 = param.shape[0]
    param_len_1 = param.shape[1]

    channels_combined_list = []
    for inx, channel in enumerate(inp[0]):

        channel_pad_list = []
        # Pad columns (axis 1) in each channel
        for column in channel.T:
            channel_pad = _pad_ld(column, param_len_0)
            channel_pad_list.append(channel_pad)
        channel_pad_real = np.array(channel_pad_list).T

        channel_pad_list = []
        # Pad rows (axis 0) after columns padded
        for row in channel_pad_real:
            channel_pad = _pad_ld(row, param_len_1)
            channel_pad_list.append(channel_pad)

        channel_pad = np.array(channel_pad_list)
        channels_combined_list.append(channel_pad)

    channels_combined = np.stack(channels_combined_list)

    samples = inp
    # Pad all samples with the padded example from first sample's first channel
    padded_batch = U.channels_pad_batch(samples=samples, example=channels_combined[0])

    return padded_batch

   


         



    
    
    








 




































































































