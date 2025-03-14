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
        output, input_pad, raw_output = conv_ld(inp, conv_layer.param, conv_layer.jump)
        conv_layer.input_pad = input_pad
        conv_layer.kernels = get_kernels(conv_layer.param, input_pad)
    
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

def conv_ld(inp: ndarray, param: ndarray, jump: int = 0) -> ndarray:
    
    # initilization of entry data
    input_pad  = input_pad_calc(inp, param)
    print(input_pad)
    # quit()
 
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
        
     
        out_array = mask * param
        out_list_computed = np.sum(out_array)

        out_array_computed2[row_inx, column_inx] = np.sum(out_array)
        out_array2[(row_inx + (param_in_row * column_inx ))] = out_array
     
    
    return out_array_computed2, input_pad, out_array2


def conv_ld_sum(inp: ndarray, param: ndarray) -> ndarray:

    out, input_pad = conv_ld(inp, param)
    
    return np.sum(out)
# HEERE KERNELS NEED REAPIR YESYESYESYESYES 28.02.2025 28.02.202528.02.202528.02.202528.02.202528.02.202528.02.202528.02.202528.02.202528.02.2025
def get_kernels(param: ndarray, input_pad: ndarray) -> ndarray:
    #  for 2d data it returns 4d data

    # input_pad = np.atleast_2d(input_pad)
    # kernels = np.zeros((input_pad.shape[0], input_pad.shape[1] - (param.shape[0] - 1), param.shape[0]))
    # for inx, row in enumerate(input_pad):
    #   kernel = np.zeros((row.shape[0] - (param.shape[0] - 1), param.shape[0]))
      
    #   for i in range(row.shape[0] - (param.shape[0] - 1)):
    #       kernel[i] = row[i : param.shape[0] + i]
    
    #   kernels[inx] = kernel 
    #   if not 'kernels_real' in locals():
    #       kernels_real = kernel  
    #   else:
    #       kernels_real = np.concatenate([kernels_real, kernel])
    print(input_pad)
    # quit()
    # # 3d array
    # Calculating size of unpadded input data (input_pad.shape[0] - (param.shape[0] - 1), input_pad.shape[1] - (param.shape[1] - 1))
    kernels =np.zeros((input_pad.shape[0] - (param.shape[0] - 1), input_pad.shape[1] - (param.shape[1] - 1), param.shape[0], param.shape[1]))
    # quit()
    for column_inx, column in enumerate(input_pad.T):
      for row_inx, row in enumerate(input_pad):
        mask = input_pad[row_inx : param.shape[0] + row_inx, column_inx : param.shape[1] + column_inx]
        if mask.shape[0] != param.shape[0] or mask.shape[1] != param.shape[1]:
            break
        
        
        # quit()
        kernels[row_inx, column_inx] = mask
        
        

    # quit()
    
    return kernels


# CURRENTLY GET KERNELS GIVES 2D ARRAY WHERE SHAPE[0] ARE ROWS OF DATA IDK IF I WONT THAT OR SIMPLE 1D FOR KERNEL BUT WE WILL SE I GUEES
# HEREHREHREHRE 1.03.2024
# PROBLEM 05.03.2025 JAK SZUKAC KTORE IMPUTY Z KTORYMI WAGAMI SA POWIAZANE W 2 WYMIAROWYM INPUCIE WKONCU UZYWAMY MASKI A NIE ZWYKLEJ ITERACJI
def map_input_weight_matrix(inp: ndarray, param: ndarray, input_pad: ndarray, kernels: ndarray, weights: ndarray, map: str) -> ndarray:
     
    
    # turning kernels back to 2d
    # print(kernels)
    # kernels = kernels.reshape(((kernels.shape[0] * kernels.shape[1], kernels.shape[2] * kernels.shape[3])))
    weight_index = {}
    # Searching for same index in a kernel
    # print(input_pad)
    # print(kernels[0][0])
    # print(kernels.shape)
    weights_map = {}
    ouuuut = 0
    for column_inx, column in enumerate(input_pad.T):
    #   print('BREAKKK')
      for row_inx, row in enumerate(input_pad):
          mask = input_pad[row_inx : param.shape[0] + row_inx, column_inx : param.shape[1] + column_inx]
          if mask.size != param.size:
             break
        #   print(mask)
        #   print(kernels_column)
        #   print(mask)
        #   print(input_pad)
        #   quit()
          for row_mask_inx, row_mask in enumerate(mask):
            # print(row_mask)   
            
            for column_mask_inx, column_mask in enumerate(row_mask):
               ouuuut += 1
            #    print(i)
               
            #    print((row_mask_inx + row_inx), (column_mask_inx + column_inx))
            #    print(f'mask.size: {mask.size}')
            #    print(f'column_mask_inx: {column_mask_inx}')
            #    print(f'row_mask_inx: {row_mask_inx}')
            #    print(f'column_inx: {column_inx}')
            #    print(f'row_inx: {row_inx}')
            #    print(f'column.shape: {column.shape[0]}')
            #    print(f'mask.shape[0]: {mask.shape[0]}')
            #    print(f'row_mask.size: {row_mask.size}')
            #    IT SEEMS TO WORK FOR THE FRIST COLUMN_INX ITERATION ON SECOND NO THO
               weight_num = ((column_mask_inx + 1) + row_mask.size * row_mask_inx) + mask.size * (column_inx * (column.shape[0] - (mask.shape[0] - 1))) + (row_inx * mask.size)
               weight_index = ((weight_num - 1) // weights.shape[1], (weight_num - 1) - (weight_num - 1) // weights.shape[1] * weights.shape[1])
               try:
                 weights_map[row_mask_inx + row_inx, column_mask_inx + column_inx].append(weight_index)
               except KeyError:
                 weights_map[row_mask_inx + row_inx, column_mask_inx + column_inx] = [weight_index]
         

    # print(weights_map)
    # print(weights.shape)
    # quit()
    return weights_map

def input_deriative(inp: ndarray, input_pad: ndarray, weight_index: map_input_weight_matrix, weights: ndarray) -> ndarray:
    
    
    
    input_gradients_list = []
    for inx_row, row in enumerate(input_pad):
      input_gradients = np.zeros(row.shape)
      for index_column in np.ndindex(row.shape):
        #   print(weight_index)
         
         
          weights_indexes = weight_index[inx_row,  index_column[0]]
          # getting weights conntected to input we work with currently
          inputs_weights = [weights[*i] for i in weights_indexes]
      
         
          #  here gradient of kernel is one because we are only adding them and its fairly simple
          gradient = np.sum(inputs_weights)
          input_gradients[*index_column] = gradient
    
      
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
