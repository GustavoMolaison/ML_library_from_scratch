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
            # setting back to 2d cause get kernels returns 3d
            self.kernels = self.kernels.reshape(self.kernels.shape[0] * self.kernels.shape[1], self.kernels.shape[2])
         
            # self.kernels_weights = np.random.uniform(0, 1, (self.kernels.shape[0], self.param.shape[0]))
            self.kernels_weights = np.ones((self.kernels.shape[0] * self.kernels.shape[1], self.param.shape[0]))
        def get_info(self):
            print(f'self.kernels{self.kernels.shape}')
            print(f'self.kernels_weights{self.kernels_weights.shape}')

    def forward_conv(self, inp: ndarray, conv_layer: conv_layer) -> ndarray:
        if inp.ndim != conv_layer.inp.ndim:
            print('___________________________ERROR___________________________ \n Dimension inconsencty, (forward_conv function)')
            quit()
        output, input_pad, raw_output = conv_ld(inp, conv_layer.param, conv_layer.jump)
        conv_layer.input_pad = input_pad
        if output.ndim != conv_layer.inp.ndim:
            print('___________________________ERROR___________________________ \n Dimension inconsencty2~-input_pad, (forward_conv function)')
            quit()
        return output, input_pad
    
    def output_sum_basic_ver(self, inp: ndarray) -> ndarray:
        return np.sum(inp)
    
    def backward_conv(self, inp: ndarray, conv_layer: conv_layer) -> ndarray:
        if inp.ndim != conv_layer.inp.ndim:
            print('___________________________ERROR___________________________ \n Dimension inconsencty, (Backward_conv function)')
            quit()
        weight_index = map_input_weight_matrix(inp, conv_layer.param, conv_layer.input_pad, map = 'weight')
        input_index = map_input_weight_matrix(inp, conv_layer.param, conv_layer.input_pad, map = 'input')
        i_der = input_deriative(inp, conv_layer.input_pad, weight_index, conv_layer.kernels_weights)
        w_der = weight_deriative(inp, conv_layer.input_pad, input_index, conv_layer.kernels_weights)
        return i_der, w_der






def _pad_ld(inp: ndarray, num: int) -> ndarray:
    # add 0 times num to begging and end of an array
    z = np.array([0])
    z =  np.repeat(z, num)

    if inp.ndim == 1:
       return np.concatenate([z, inp, z])
    if inp.ndim == 2:
       return np.stack([np.concatenate([z, inp[0], z]) for i in range(inp.shape[0])])


# inp = input param = filter
def input_pad_calc(inp: ndarray, param: ndarray, jump: int = 0) -> ndarray:
     # filling entry data
    param_len = param.shape[0]
    param_mid = param_len // 2
 
    
    input_pad = _pad_ld(inp, param_mid)
    
    if not jump == 0:
       amount_of_param_passes = input_pad.shape[0] - (param_len - 1)
       confirmed_iteration =  (1 + ((amount_of_param_passes - 1) // (jump + 1)) )
       skipped_iteration =  amount_of_param_passes - confirmed_iteration
 
       input_pad = _pad_ld(input_pad, (skipped_iteration + (jump - 1)))

    return input_pad
# class indexed_data():
#         def __init__(self):
         
def kernel_forward(inp: ndarray, param: ndarray, input_pad: ndarray, jump: int = 0) -> ndarray:

    jump_calc = 0
    for o in range(inp.shape[0]):
        for p in range(param.shape[0]):
            inp[o] += param[p] * input_pad[o+p + jump_calc]
            
        jump_calc += jump   


             
      


    
    return inp, input_pad

def conv_ld(inp: ndarray, param: ndarray, jump: int = 0) -> ndarray:
    
    # initilization of entry data
    
    input_pad_list = []
    for column in inp.T:
        input_pad = input_pad_calc(column, param, jump)
        input_pad_list.append(input_pad)

    input_pad_real = np.array(input_pad_list).T
    input_pad_list = []
   
    for row in input_pad_real:
        input_pad = input_pad_calc(row, param, jump)
        input_pad_list.append(input_pad)

    input_pad = np.array(input_pad_list)
    
   
    
    param_in_row = (input_pad.shape[0] - (param.shape[0] - 1))
    param_in_columns =  (input_pad.shape[1] - (param.shape[1] - 1))
    out_array2 = np.zeros((param_in_row * param_in_columns, param.shape[0], param.shape[1]))
    out_array_computed2 = np.zeros(inp.shape)
    
    for column_inx, column in enumerate(input_pad.T):
      for row_inx, row in enumerate(input_pad):
        kernel_count = 0
        mask = input_pad[row_inx : param.shape[0] + row_inx, column_inx : param.shape[1] + column_inx]

        if mask.shape[0] != param.shape[0] or mask.shape[1] != param.shape[1]:
            break
        # if 'out_array' in locals() and 'out_list_computed' in locals():
        #     out_array2[(row_inx + (param_in_row * column_inx ))] = out_array
        #     out_list_computed2.append(np.sum(out_list_computed))

        # out_array = np.zeros(param.shape)
        # out_list_computed = []
        
     
        out_array = mask * param
        out_list_computed = np.sum(out_array)

        out_array_computed2[row_inx, column_inx] = np.sum(out_array)
        out_array2[(row_inx + (param_in_row * column_inx ))] = out_array
       
        # quit()
        # for row_mask_inx, row_mask in enumerate(mask):
        #     out = np.zeros(row_mask.shape)
        #     jump_calc = 0
        #     kernel = param[kernel_count]
        #     kernel_count += 1
        #     # print(f'{input_pad}\n\n mask')
        #     # print(f'{mask}\n\n')
        #     # print(f'{mask}\n\n kernel')
        #     # print(f'{kernel}\n\n row_mask')
        #     # print(f'{row_mask}\n\n out')
        #     # print(f'{out}\n')
        #     # quit()
        #     for p in range(kernel.shape[0]):
        #             out[p] = kernel[p] * row_mask[p + jump_calc]
            
        #     jump_calc += jump
        #     out_array[row_mask_inx] = out
            
        #     out_list_computed.append(np.sum(out))
        
    

  
       
    # out_array2[(param_in_row - 1) + (param_in_row * (param_in_columns - 1) )] = out_array
    # out_list_computed2.append(np.sum(out_list_computed))  
    # print(out_array2)  
    # print((out_list_computed2))
    # print(out_array_computed2)
    # quit()
        
    
    return out_array_computed2, input_pad_real, out_array2


def conv_ld_sum(inp: ndarray, param: ndarray) -> ndarray:

    out, input_pad = conv_ld(inp, param)
    
    return np.sum(out)

def get_kernels(param: ndarray, input_pad: ndarray) -> ndarray:

    input_pad = np.atleast_2d(input_pad)
    kernels = np.zeros((input_pad.shape[0], input_pad.shape[1] - (param.shape[0] - 1), param.shape[0]))
    for inx, row in enumerate(input_pad):
      kernel = np.zeros((row.shape[0] - (param.shape[0] - 1), param.shape[0]))
      
      for i in range(row.shape[0] - (param.shape[0] - 1)):
          kernel[i] = row[i : param.shape[0] + i]
    
      kernels[inx] = kernel 
      if not 'kernels_real' in locals():
          kernels_real = kernel  
      else:
          kernels_real = np.concatenate([kernels_real, kernel])
    
    # 3d array
    return kernels


# CURRENTLY GET KERNELS GIVES 2D ARRAY WHERE SHAPE[0] ARE ROWS OF DATA IDK IF I WONT THAT OR SIMPLE 1D FOR KERNEL BUT WE WILL SE I GUEES
def map_input_weight_matrix(inp: ndarray, param: ndarray, input_pad: ndarray, map: str) -> ndarray:
     
    kernels = get_kernels(param, input_pad)
    input_index = {}
    weight_index = {}
    # Searching for same index in a kernel
    for inx, row in enumerate(input_pad):
      
    
      for inx2, kernel in enumerate(kernels[inx]):
         
          
       
          #   c=1  
          # np index gets all indexes from array 
          for index in np.ndindex(row.shape):
              
              # we are gonna look for this index in our kernel
              # checking wether index we look for isnt to big to exist in our kernel
              if index[0] > (len(kernel) - 1) + inx2:
                  break
              # iterating over every index isnise our current kernel to compare it to input we look
              
              for k_value_index in np.ndindex(kernel.shape):
                  # that how we calculate if the index and our kernel_value is the same  excact number at the same excact index
                  if k_value_index[0] + inx2 == index[0]:
                    #  print(f'found{index}')
                    #  if c % 3 == 0:
                    #      print('\n')
                    #  c+= 1
                  #  And saving indexs of location of our inputs inside out weights matrix
                     if map == 'weight':
                       try:
                            input_index[f'input{inx, index[0]}'].append([inx * kernels.shape[1] + inx2, *k_value_index])
                            # print('APPENDING')
                       except KeyError:
                            input_index[f'input{inx, index[0]}'] = [[inx * kernels.shape[1] + inx2, *k_value_index]]
                            # print('CREATING')
                     if map == 'input':
                       try:
                            input_index[f'weight{[inx * kernels.shape[1] + inx2, *k_value_index]}'].append([inx, index[0]])
                            # print('APPENDING')
                       except KeyError:
                            input_index[f'weight{[inx * kernels.shape[1] + inx2, *k_value_index]}'] = [[inx, index[0]]]
                            # print('CREATING') 
      
    
    
    return input_index

def input_deriative(inp: ndarray, input_pad: ndarray, weight_index: map_input_weight_matrix, weights: ndarray) -> ndarray:
    
    
    
    input_gradients_list = []
    for inx, row in enumerate(input_pad):
      input_gradients = np.zeros(row.shape)
      for index in np.ndindex(row.shape):
          print(weight_index)
         
         
          weights_indexes = weight_index[f'input{inx,  index[0]}']
          # getting weights conntected to input we work with currently
          inputs_weights = [weights[*i] for i in weights_indexes]
      
         
          #  here gradient of kernel is one because we are only adding them and its fairly simple
          gradient = np.sum(inputs_weights)
          input_gradients[*index] = gradient
    
      
      input_gradients_list.append(input_gradients)  
    
    input_gradients_real = np.array(input_gradients_list)
    # print(input_gradients_real)
    # quit()
    return input_gradients_real

def weight_deriative(inp: ndarray, input_pad: ndarray, input_index: map_input_weight_matrix, weights: ndarray)  -> ndarray:
    for row in input_pad:
      weight_gradients = np.zeros(weights.shape)

      for index in np.ndindex(weights.shape):
        #   print(input_index)
        #   print(*index)
        #   print(input_pad)
        #   quit()
          input_indexes =  input_index[f'weight{[*index]}']
        #   print(input_pad)
        #   print(input_pad[input_indexes[0]])
        #   quit()
          weights_inputs = [input_pad[i[0], i[1]] for i in input_indexes]


          gradient = np.sum(weights_inputs)
          weight_gradients[*index] = gradient
      
      if not 'weight_gradients_real' in locals():
          weight_gradients_real = weight_gradients  
      else:
          weight_gradients_real = np.concatenate([weight_gradients_real, weight_gradients])
    
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


input_1d = np.array([[1,2,3,4,5],
                     [5,2,3,4,5],
                     [5,2,3,4,5],
                     [5,2,3,4,5]])
# input_1d = np.array([1,2,3,4,5])
param_1d = np.array([[2,1,1],
                     [1,1,1],
                     [1,1,2]])

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
print(f'sum{sum}')
print(f'input_der{i_der}')
print(f'weight_der{w_der}')
