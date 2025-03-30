import numpy as np
from numpy import ndarray

class convulsive_model():
    def __init__(self):
        
        pass

    def initilialization(self,  inp: ndarray, param: ndarray, jump: int = 0 ):
        self.inp = inp
        self.param = param
        self.jump = jump

    class conv_layer():
        def __init__(self, inp: ndarray, param: ndarray, jump: int = 0 ):
            self.inp = inp
            self.param = param
            self.jump = jump
            # creating empty data we only wont shapes here altghou function needs actual arrays
            inp_shape = np.zeros(self.inp.shape)
            param_shape = np.zeros(param.shape)
            # getting data after adding 0
            input_pad_shape = input_pad_calc(inp_shape, param_shape)
            
            
            self.kernels = get_kernels(param_shape, input_pad_shape)

            # self.kernels_weights = np.random.uniform(0, 1, (self.kernels.shape[0], self.param.shape[0]))
            self.kernels_weights = np.ones((self.kernels.shape[0] * self.kernels.shape[1], self.kernels.shape[2] * self.kernels.shape[3]))
            print(self.kernels_weights.shape)
            
        
        def get_info(self):
            print(f'self.kernels{self.kernels.shape}')
            print(f'self.kernels_weights{self.kernels_weights.shape}')

    def forward_conv(self, inp: ndarray, conv_layer: conv_layer) -> ndarray:
        if inp.ndim != conv_layer.inp.ndim:
            print('___________________________ERROR___________________________ \n Dimension inconsencty, (forward_conv function)')
            quit()
        output, input_pad, conv_layer.kernels_data = conv_ld(inp, conv_layer.param, conv_layer.jump)
        conv_layer.input_pad = input_pad
        conv_layer.kernels = get_kernels(conv_layer.param, input_pad)
        
        
    
        if output.ndim != conv_layer.inp.ndim:
            print('___________________________ERROR___________________________ \n Dimension inconsencty2~-input_pad, (forward_conv function)')
            print(f'Desired dimension of output: {conv_layer.inp.ndim}, actual: {output.ndim}')
            print()
            quit()
        return output, input_pad
    
    def output_sum_basic_ver(self, inp: ndarray) -> ndarray:
        return np.sum(inp)
    
    def backward_conv(self, inp: ndarray, conv_layer: conv_layer) -> ndarray:
        if inp.ndim != conv_layer.inp.ndim:
            print('___________________________ERROR___________________________ \n Dimension inconsencty, (Backward_conv function)')
            quit()
        weight_index = map_input_weight_matrix(inp, conv_layer.param, conv_layer.input_pad, conv_layer.kernels, conv_layer.kernels_weights, map = 'weight')
        input_index = map_input_weight_matrix(inp, conv_layer.param, conv_layer.input_pad, conv_layer.kernels, conv_layer.kernels_weights,  map = 'input')
        i_der = input_deriative(inp, conv_layer.input_pad, weight_index, conv_layer.kernels_weights)
        w_der = weight_deriative(inp, conv_layer.input_pad, input_index, conv_layer.kernels_weights)
        return i_der, w_der






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
    for inx, channel in enumerate(inp):
        print('HALOOOOOOOOOOOOOOO')
        
        
        

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
    
    return channels_combined

   


         
def kernel_forward(inp: ndarray, param: ndarray, input_pad: ndarray, jump: int = 0) -> ndarray:

    jump_calc = 0
    for o in range(inp.shape[0]):
        for p in range(param.shape[0]):
            inp[o] += param[p] * input_pad[o+p + jump_calc]
            
        jump_calc += jump   


             
      


    
    return inp, input_pad

# PURPOUSE#######
# CHECK? -> WORKING V
# Calculating singe outputs from padded input using masks also saving all used masks/kernels
def conv_ld(inp: ndarray, param: ndarray, jump: int = 0) -> ndarray:
    
    # initilization of entry data
    input_pad  = input_pad_calc(inp, param)
    channels_amount = inp.shape[0]
#   "+ input_pad.ndim" is making sure code works with multiple channels and singe channel so its just shape[].
#   Two rows below calculate how many kernels will fit in shape[0] and [1].
    param_in_row = (input_pad.shape[(-2 + input_pad.ndim)] - (param.shape[0] - 1))
    param_in_columns =  (input_pad.shape[(-1 + input_pad.ndim)] - (param.shape[1] - 1))
    # Calculating how many kernels will be in our data.
    # Param in row * param in columns is total sum of kernels we gonna need, shapes are just well, shapes of kernel.
    kernels = np.zeros((channels_amount, param_in_row * param_in_columns, param.shape[0], param.shape[1]))
    # Here are stored channels cause each will be calculated seperatly.
    # Note that kernels are saved all in once. 
    channels_combined = []
    # We are looping for each channel both padded and unpadded.
    # Note that kernels are saved all in once its easier than nesting another loop.
    # To clarify we loop for channels in terms of input, kernels are saved all in once this is standard in this code.
    for channel_idx, (channel, out_array_computed) in enumerate(zip(input_pad, np.zeros(inp.shape))):
    # Looping through columns and rows inside singe channel.
    # Frist we move downward icreasing rows.
    # Then we move to the next column.
      for column_inx, column in enumerate(channel.T):
        for row_inx, row in enumerate(channel):
        # Calculating mask for our padded data it moves alongside rows then change columns.
          mask = channel[row_inx : param.shape[0] + row_inx, column_inx : param.shape[1] + column_inx]
        # Because sizes are not precalculated we check if the shape of param is diffrent of that of the mask.
        # If true there is no more space for mask to move inside current column so we go to another using break.
          if mask.shape[0] != param.shape[0] or mask.shape[1] != param.shape[1]:
              break
        # Current mask (so part of input data) is multiplied by kernel/param then summed to get single output. 
          out_array = mask * param
          out_list_computed = np.sum(out_array)
        
        # Assiging single output to a index in singe channel.
          out_array_computed[row_inx, column_inx] = np.sum(out_array)
        # Assigin whole kernel before summing to an index by order column then rows.
          kernels[channel_idx, (row_inx + (param_in_row * column_inx ))] = out_array
    # Adding fully proccesed channel to list and moving to the next.
      channels_combined.append(out_array_computed) 
    # Combining all saved channels.
    channels_combined = np.stack(channels_combined)
    
    
    return channels_combined, input_pad, kernels


def conv_ld_sum(inp: ndarray, param: ndarray) -> ndarray:

    out, input_pad = conv_ld(inp, param)
    
    return np.sum(out)
# purpouse 
# gets shape of kernels (data full of 0) based on input_pad and param
def get_kernels(param: ndarray, input_pad: ndarray) -> ndarray:
    #  for 2d data it returns 4d data

   
    kernels_combined = []

    for channel in input_pad:
      # # 3d array
      # Calculating size of unpadded input data (input_pad.shape[0] - (param.shape[0] - 1), input_pad.shape[1] - (param.shape[1] - 1))
      kernels =np.zeros((channel.shape[0] - (param.shape[0] - 1), channel.shape[1] - (param.shape[1] - 1), param.shape[0], param.shape[1]))
      
      for column_inx, column in enumerate(channel.T):
        for row_inx, row in enumerate(channel):
          mask = channel[row_inx : param.shape[0] + row_inx, column_inx : param.shape[1] + column_inx]
          if mask.shape[0] != param.shape[0] or mask.shape[1] != param.shape[1]:
              break
        
          kernels[row_inx, column_inx] = mask
          
      kernels_combined.append(kernels)
   
    kernels_combined  = np.stack(kernels_combined)
    
    
    return kernels_combined



# PROBLEM 05.03.2025 JAK SZUKAC KTORE IMPUTY Z KTORYMI WAGAMI SA POWIAZANE W 2 WYMIAROWYM INPUCIE WKONCU UZYWAMY MASKI A NIE ZWYKLEJ ITERACJI
def map_input_weight_matrix(inp: ndarray, param: ndarray, input_pad: ndarray, kernels: ndarray, weights: ndarray, map: str) -> ndarray:
     
    
    # turning kernels back to 2d
    # print(kernels)
    # kernels = kernels.reshape(((kernels.shape[0] * kernels.shape[1], kernels.shape[2] * kernels.shape[3])))
    # Searching for same index in a kernel
    # print(input_pad)
    # print(kernels[0][0])
    # print(kernels.shape)
    channels_combined = []
    for channel in input_pad:
      weight_index = {}
      weights_map = {}
      ouuuut = 0
      for column_inx, column in enumerate(channel.T):
      #   print('BREAKKK')
        for row_inx, row in enumerate(channel):
            mask = channel[row_inx : param.shape[0] + row_inx, column_inx : param.shape[1] + column_inx]
            if mask.size != param.size:
                break
            #   print(mask)
            #   print(kernels_column)
            #   print(mask)
            #   print(channel)
            #   quit()
            for row_mask_inx, row_mask in enumerate(mask):
                # print(row_mask)   
            
                for column_mask_inx, column_mask in enumerate(row_mask):
                    ouuuut += 1
             
                    print((row_mask_inx + row_inx), (column_mask_inx + column_inx))
                    print(f'mask.size: {mask.size}')
                    print(f'column_mask_inx: {column_mask_inx}')
                    print(f'row_mask_inx: {row_mask_inx}')
                    print(f'column_inx: {column_inx}')
                    print(f'row_inx: {row_inx}')
                    print(f'column.shape: {column.shape[0]}')
                    print(f'row.shape: {row.shape[0]}')
                    print(f'mask.shape[0]: {mask.shape[0]}')
                    print(f'row_mask.size: {row_mask.size}')
        
                weight_num = ((column_mask_inx + 1) + row_mask.size * row_mask_inx) + mask.size * (column_inx * (column.shape[0] - (mask.shape[0] - 1))) + (row_inx * mask.size)
                weight_index = ((weight_num - 1) // weights.shape[1], (weight_num - 1) - (weight_num - 1) // weights.shape[1] * weights.shape[1])
                try:
                    weights_map[row_mask_inx + row_inx, column_mask_inx + column_inx].append(weight_index)
                except KeyError:
                    weights_map[row_mask_inx + row_inx, column_mask_inx + column_inx] = [weight_index]
            print('HAWKTUAHWTUAH TUAHWTUAHHAWKTUAHWTUAH TUAHWTUAHHAWKTUAHWTUAH TUAHWTUAHHAWKTUAHWTUAH TUAHWTUAHHAWKTUAHWTUAH TUAHWTUAHHAWKTUAHWTUAH TUAHWTUAHHAWKTUAHWTUAH TUAHWTUAHHAWKTUAHWTUAH TUAHWTUAHHAWKTUAHWTUAH TUAHWTUAHHAWKTUAHWTUAH TUAHWTUAHHAWKTUAHWTUAH TUAHWTUAHHAWKTUAHWTUAH TUAHWTUAHHAWKTUAHWTUAH TUAHWTUAH')
            quit()
      channels_combined.append(weights_map) 
#   currently savinf using list is it valid option thoug?
    # print(channels_combined)
    print(weights_map)
    quit()
    weights_map = channels_combined
    return weights_map

def input_deriative(inp: ndarray, input_pad: ndarray, weight_index: map_input_weight_matrix, weights: ndarray) -> ndarray:
    
    
    # print(weight_index)
    
    
    channels_combined = np.zeros(input_pad.shape)
    for channel_idx, (channel_weight_index, channel_input) in enumerate(zip(weight_index, input_pad)):
      input_gradients_list = []
      for inx_row, row in enumerate(channel_input):
        input_gradients = np.zeros(row.shape)
        for index_column in np.ndindex(row.shape):
          #   print(channel_index)
            print(channel_weight_index)
            quit()
         
            print(channel_weight_index) 
            weights_indexes = channel_weight_index[inx_row,  index_column[0]]
            # getting weights conntected to input we work with currently
            inputs_weights = [weights[*i] for i in weights_indexes]
      
         
            #  here gradient of kernel is one because we are only adding them and its fairly simple
            gradient = np.sum(inputs_weights)
            input_gradients[*index_column] = gradient
    

        input_gradients_list.append(input_gradients) 

      channels_combined[channel_idx] = np.array(input_gradients_list)
    #   input_gradients_real = np.array(input_gradients_list)
      # print(input_gradients_real)
      # quit()
      input_gradients_real = channels_combined
      return input_gradients_real

def weight_deriative(inp: ndarray, input_pad: ndarray, input_index: map_input_weight_matrix, weights: ndarray)  -> ndarray:
    for row in input_pad:
      weight_gradients = np.zeros(weights.shape)

      for index in np.ndindex(weights.shape):
        #   print(input_index)
        #   quit()
        #   print(input_index)
        #   print(*index)
        #   print(index)
        #   quit()
          input_indexes = [key for key, value in input_index.items()
                           if any(value_item == index for value_item in value)]
                        #    for value_item in value if value_item == index]
        #   print(input_indexes)
        #   print(index)
        #   input_indexes =  input_index[f'weight{[*index]}']
        #   print(input_pad)
        #   print(input_pad[input_indexes[0]])
        #   quit()
          weights_inputs = [input_pad[row, column] for row, column in input_indexes]

        #   Here we are just adding inputs cause our operation is basiclt weight x input (for now we dont work with dense network at all)
          gradient = np.sum(weights_inputs)
          weight_gradients[*index] = gradient
      
      if not 'weight_gradients_real' in locals():
          weight_gradients_real = weight_gradients  
      else:
          weight_gradients_real = np.concatenate([weight_gradients_real, weight_gradients])
    
    # print(weight_gradients)
    # quit()
    return weight_gradients

        


    
   
            # if index == np.array([i.index() + kernels.index(kernel) for i in kernel ]).any():

    # print(kernels)

    # for i in input_pad:


def np_index(arr, value):
    result = np.where(arr == value)[0]  # Get all indexes
    if result.size > 0:  # Check if the value exists
        return int(result[0])  # Return first occurrence
    else:
        raise ValueError(f"{value} is not in array")


# input_1d = np.array([[1,2,3,4,5],
#                      [5,2,3,4,5],
#                      [5,2,3,4,5],
#                      [5,2,3,4,5]])

input_1d = np.array([[[1,2,3,4,5],
                     [5,2,3,4,5],
                     [5,2,3,4,5],
                     [5,2,3,4,5]],

                     [[1,2,3,4,5],
                     [5,2,3,4,5],
                     [5,2,3,4,5],
                     [5,2,3,4,5]],

                     [[1,2,3,4,5],
                     [5,2,3,4,5],
                     [5,2,3,4,5],
                     [5,2,3,4,5]]])
# input_1d = np.array([1,2,3,4,5])
# param_1d = np.array([[1,1,1],
#                      [1,1,1],
#                      [1,1,1]])
param_1d = np.array([[1,1,1],
                     [1,1,1],
                     [1,1,1]
                     ])

# input, pad_inp = conv_ld(input_1d, param_1d)
# x = map_input_weight_matrix(input, param_1d, pad_inp, map = 'input')
# print(x)
# quit()

model = convulsive_model()
conv_1 = model.conv_layer(inp = input_1d, param = param_1d, jump = 0)
output, pad_input = model.forward_conv(input_1d, conv_1)
sum = model.output_sum_basic_ver(output)
i_der, w_der = model.backward_conv(output, conv_1)
print(f'output{output}')
print(f'input_der{i_der}')
print(f'weight_der{w_der}')
print(f'sum: {sum}')

# NOTATKI
# PROBLEM JES W MAPOWANIU JAK COS POPROSTU NEI DAJE ERRORA