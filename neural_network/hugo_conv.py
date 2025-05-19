import numpy as np
from numpy import ndarray
from numpy.lib.stride_tricks import sliding_window_view
from hugo_utility import Utility as U 

class Conv_layer():
        def __init__(self, model = None):
           self.model = model
           self.layer_set = False


        def set_layer(self, param: ndarray, activaion_function = None, weight_initialization = None, jump: int = 0):
            self.param = param
            self.jump = jump
            self.activation_function = activaion_function
            self.activation_functions = {'none': U.no_activation_function, 'sigmoid' : U.sigmoid, 'relu' : U.relu, 'leaky relu': U.leaky_relu, 'tanh': U.tanh}
            self.weight_initialization = weight_initialization
            self.weights_initializations = {'linear' : U.linear, 'he' : U.he, 'xavier': U.xavier}
            self.layer_set = True
                 
           
        def forward_L(self, input):
           if self.layer_set != True:
              print('Conv_layer is not set. Use "Conv_layer.set_layer" before anything else!')
              raise LookupError()
           self.input = input
           self.output = np.zeros(self.input.shape)
           self.input_grads = np.zeros(self.input.shape)
           self.weight_grads = []
           for sample_idx, sample in enumerate(self.input):
              # Calculating new values after aplying kernels to padded data along with kernels and input pad
              output, self.input_pad, self.kernels = conv_ld(sample, self.param, self.jump)
              # weights sorted only by channels (3, number of weights for channel)
              self.kernels_weights = np.reshape(self.kernels, (self.kernels.shape[0],  self.kernels.shape[1] *  self.kernels.shape[2] * self.kernels.shape[3] * self.kernels.shape[4]))
                                                                       #Pre_backward
              # mapping to what input each weight is connected(returns dict)
              self.weight_index = map_input_weight_matrix(sample, self.param, self.input_pad, self.kernels, self.kernels_weights, map = 'weight')
              # mapping to what weight each input is connected(returns dict)
              self.input_index = map_input_weight_matrix(sample, self.param, self.input_pad, self.kernels, self.kernels_weights,  map = 'input')
              # calculates how each input influence the output
              input_grad = input_derivative(sample, self.input_pad, self.weight_index, self.kernels_weights, self.param)
              # calculates how each weight thus param influence the output
              weight_grad = weight_derivative(sample, self.input_pad, self.input_index, self.kernels_weights)
              
              self.output[sample_idx] = output
              self.input_grads[sample_idx] = input_grad
              self.weight_grads.append(weight_grad)
         
           self.weight_grads = np.stack(self.weight_grads)
          
           flatten = np.reshape(self.output, (self.output.shape[0], -1))
            
           return flatten

        def backward_L(self, grad):
           if self.layer_set != True:
              print('Conv_layer is not set. Use "Conv_layer.set_layer" before anything else!')
              raise LookupError()
           grad = grad * self.af_gradient

           layer_weight_grad = np.dot(self.weight_grad.T, grad)

           layer_input_grad = np.dot(grad, self.layer_weights.T)
           
           
        
        def get_info(self):
            print(f'self.kernels{self.kernels.shape}')
            print(f'self.kernels_weights{self.kernels_weights.shape}')








# changes array from (180) to (3, 60) [180 = input, 60 = param]
def d1_to_d2(input, param):
    window = param.size
    output = np.zeros((input.shape[0], input.shape[1] // window, *param.shape))
    for channel_idx, channel in enumerate(input):
        channel_w_der = np.zeros((input.shape[1] // window, *param.shape))
    
        for stride in range((input.shape[1] // window)):
            channel_w_der[stride] = channel[stride * 9 : window + stride * 9].reshape(param.shape)
        output[channel_idx] = channel_w_der
   
    return output
       
       
def unpad_info(inp, param, input_pad):
    z_to_add_0 = inp.size - (inp.size - (param - 1))
    index_to_skip = []
    diff = 0
    for i in range(z_to_add_0):
       if i % 2 == 0:
         index_to_skip.append(i - diff)
         diff += 1
       if i % 2 == 1:
         index_to_skip.append(input_pad - diff)
    if z_to_add_0 % 2 == 0:
       index_to_add = z_to_add_0 //2
    if z_to_add_0 % 2 == 1:
       index_to_add = z_to_add_0 //2
    return index_to_skip

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
    # print(inp.shape)
    # quit()
    for inx, channel in enumerate(inp):
        
        
        
        

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
    
    input_pad = input_pad_calc(inp, param)
    # axis 0 is skipped because those are channels 
    patches = sliding_window_view(input_pad, (param.shape[0], param.shape[1]), axis = (1, 2))
    

    output = np.einsum('cijxy,xy->cij', patches, param)

    kernels = patches * param
    

    return output, input_pad, kernels





                                          # OLD ORIGINAL SLOW IMPLEMENTATION
    # initilization of entry data
#     input_pad  = input_pad_calc(inp, param)
#     channels_amount = inp.shape[0]
# #   "+ input_pad.ndim" is making sure code works with multiple channels and singe channel so its just shape[].
# #   These two rows below calculates how many kernels will fit in shape[0] and [1].
#     param_in_row = (input_pad.shape[(-2 + input_pad.ndim)] - (param.shape[0] - 1))
#     param_in_columns =  (input_pad.shape[(-1 + input_pad.ndim)] - (param.shape[1] - 1))
#     # Calculating how many kernels will be in our data.
#     # Param in row * param in columns is total sum of kernels we gonna need, shapes are just well, shapes of kernel.
#     kernels = np.zeros((channels_amount, param_in_row, param_in_columns, param.shape[0], param.shape[1]))
#     # Here are stored channels cause each will be calculated seperatly.
#     # Note that kernels are saved all in once. 
#     channels_combined = []
#     # We are looping for each channel both padded and unpadded.
#     # Note that kernels are saved all in once its easier than nesting another loop.
#     # To clarify we loop for channels in terms of input, kernels are saved all in once this is standard in this code.
#     for channel_idx, (channel, out_array_computed) in enumerate(zip(input_pad, np.zeros(inp.shape))):
#     # Looping through columns and rows inside singe channel.
#     # Frist we move downward icreasing rows.
#     # Then we move to the next column.
#       for column_inx, column in enumerate(channel.T):
#         for row_inx, row in enumerate(channel):
#         # Calculating mask for our padded data it moves alongside rows then change columns.
#           mask = channel[row_inx : param.shape[0] + row_inx, column_inx : param.shape[1] + column_inx]
#         # Because sizes are not precalculated we check if the shape of param is diffrent of that of the mask.
#         # If true there is no more space for mask to move inside current column so we go to another using break.
#           if mask.shape[0] != param.shape[0] or mask.shape[1] != param.shape[1]:
#               break
#         # Current mask (so part of input data) is multiplied by kernel/param then summed to get single output. 
#           out_array = mask * param
#           out_list_computed = np.sum(out_array)
        
#         # Assiging single output to a index in singe channel.
#           out_array_computed[row_inx, column_inx] = np.sum(out_array)
#         # Assigin whole kernel before summing to an index by order column then rows.
#           kernels[channel_idx, row_inx, column_inx ] = out_array
#     # Adding fully proccesed channel to list and moving to the next.
#       channels_combined.append(out_array_computed) 
#     # Combining all saved channels.
#     channels_combined = np.stack(channels_combined)
    
    
    return output, input_pad, kernels


def conv_ld_sum(inp: ndarray, param: ndarray) -> ndarray:

    out, input_pad = conv_ld(inp, param)
    
    return np.sum(out)
# purpouse 
# gets shape of kernels (data full of 0) based on input_pad and param
def get_kernels(param: ndarray, input_pad: ndarray) -> ndarray:
    #  for 2d data it returns 4d data

    # patches = sliding_window_view(x = input_pad, window_shape = param, axis = (1, 2))
    # kernels = np.einsum('ckijxy, xy->cij', param, patches)
    kernels_combined = []

    for channel in input_pad:
      # # 3d array
      # Calculating size for our output data (input_pad.shape[0] - (param.shape[0] - 1), input_pad.shape[1] - (param.shape[1] - 1))
      kernels =np.tile(param, (channel.shape[0] - (param.shape[0] - 1), channel.shape[1] - (param.shape[1] - 1), 1, 1))
      
      for column_inx, column in enumerate(channel.T):
        for row_inx, row in enumerate(channel):
          mask = channel[row_inx : param.shape[0] + row_inx, column_inx : param.shape[1] + column_inx]
          if mask.shape[0] != param.shape[0] or mask.shape[1] != param.shape[1]:
              break
        
          kernels[row_inx, column_inx] = mask
          
      kernels_combined.append(kernels)
   
    kernels_combined  = np.stack(kernels_combined)
    
    
    
    return kernels_combined




def map_input_weight_matrix(inp: ndarray, param: ndarray, input_pad: ndarray, kernels: ndarray, weights: ndarray, map: str) -> ndarray:
     
    
    # turning kernels back to 2d
    # kernels = kernels.reshape(((kernels.shape[0] * kernels.shape[1], kernels.shape[2] * kernels.shape[3])))
    
                   # ITERATING FOR EVERY CHANNEL
    # __________________________________________________________________________________________________________________________________
    channels_combined = []
    for channel_idx, channel in enumerate(input_pad):
    #   print('channel??')
    # __________________________________________________________________________________________________________________________________


              #  CREATING VARIABLES TO SAVE DATA
    # __________________________________________________________________________________________________________________________________
      weights_map = {}  # (input index) : [(weight index)]
    # __________________________________________________________________________________________________________________________________


                        #ITERATING THROUGH AN ARRAY
    # _________________________________________________________________ _________________________________________________________________                    
    #   Frist we choose column then we iteratate through every row in this one column and move to next column so:
    #     0                   1
    #  0 [i1] <- step 1     [i4] <- step 4    step 1: i1
    #  1 [i2] <- step 2     [i5]              step 2: i2
    #  2 [i3] <- step 3     [i6]              step 3: i3
    # 
    # 
      for column_inx, column in enumerate(channel.T): # iterating for column in array
     
        for row_inx, row in enumerate(channel): # iterating for row in array
    # __________________________________________________________________________________________________________________________________
                       

                            #ITERATING WITH MASK
# __________________________________________________________________________________________________________________________________
     # row inx is row we are currently at
     # param shape[0] is size of row in mask
     # + row_inx is moving mask along the rows when 0 we start from frist row when equal to 1 we start from second so:
#    param = 2x2
     #     0    1   2
    #  0 [i1] [i4] [i7]  mask 1:  [i1] [i4] -->  mask 2: [i2] [i5]  -->   mask 3: [i3] [i6]   --> mask 4: [i4] [i7]
    #  1 [i2] [i5] [i8]           [i2] [i5]              [i3] [i6]     incorrect shape,                   [i5] [i8]
    #  2 [i3] [i6] [i9]                                                moving to next column
    #  
            mask = channel[row_inx : param.shape[0] + row_inx, column_inx : param.shape[1] + column_inx] #Calculations
            if mask.size != param.size: #incorect shape detection
                break
        #                        Here we iterate through mask we generated
        #                        We use same method as before only now
        #                        frist we move through columns, then change row.   
            for row_mask_inx, row_mask in enumerate(mask):
      
                for column_mask_inx, column_mask in enumerate(row_mask):
 # __________________________________________________________________________________________________________________________________                   
             
                    # print((row_mask_inx + row_inx), (column_mask_inx + column_inx))
                    # print(f'mask.size: {mask.size}')
                    # print(f'column_mask_inx: {column_mask_inx}')
                    # print(f'row_mask_inx: {row_mask_inx}')
                    # print(f'column_inx: {column_inx}')
                    # print(f'row_inx: {row_inx}')
                    # print(f'column.shape: {column.shape[0]}')
                    # print(f'row.shape: {row.shape[0]}')
                    # print(f'mask.shape[0]: {mask.shape[0]}')
                    # print(f'row_mask.size: {row_mask.size}')
            #                HERE IS THE ISSUE 02.04.2025     
            #                                                    Weight num is order in which weights are used durning forward function
                                                            #    So weight (0,0) will be one multiplied by input (0,0)
                    weight_num = ((column_mask_inx + 1) + row_mask.size * row_mask_inx) + mask.size * (column_inx * (column.shape[0] - (mask.shape[0] - 1))) + (row_inx * mask.size)
                    weight_index = (channel_idx, (weight_num - 1) - (weight_num - 1) // weights.shape[1] * weights.shape[1])
                    # weight_index = ((weight_num - 1) // weights.shape[1], (weight_num - 1) - (weight_num - 1) // weights.shape[1] * weights.shape[1])
                    # print(f'weight_num{weight_num}')
                    # print(f'weight_index{weight_index}')
                    # print(f'weights.shape[1]{weights.shape[1]}')
                    
                    
                    try:
                       weights_map[row_mask_inx + row_inx, column_mask_inx + column_inx].append(weight_index)
                    except KeyError:
                       weights_map[row_mask_inx + row_inx, column_mask_inx + column_inx] = [weight_index]
            
            # print(weights_map)
          
      channels_combined.append(weights_map) 

    
    
    weights_map = channels_combined
    return channels_combined

def input_derivative(inp: ndarray, input_pad: ndarray, weight_index: map_input_weight_matrix, weights: ndarray, param: ndarray) -> ndarray:
    
    #  calculating which rows and colums are padding we dont need their derivatives
    rows_to_skip = unpad_info(inp, param.shape[0], input_pad[0].shape[0])
    columns_to_skip = unpad_info(inp, param.shape[1], input_pad[0].shape[1])

    
    # Function calculates how many times mask interacted with certain input!
  
    channels_combined = np.zeros(inp.shape)
    
    for channel_idx, (channel_weight_index, channel_input) in enumerate(zip(weight_index, input_pad)):
      input_gradients = np.zeros(inp[0].shape)
      rows_skipped = 0
      columns_skipped = 0
      for inx_row, row in enumerate(channel_input):
        if inx_row in rows_to_skip:
           rows_skipped += 1
           continue
        
        for inx_column, index_column in enumerate(row):
            if inx_column in columns_to_skip:
               columns_skipped += 1
              
               continue                               
  
            weights_indexes = channel_weight_index[inx_row,  inx_column]
            inputs_weights = [weights[*i] for i in weights_indexes]
            
            
        
                  
            #  here gradient of kernel is one because we are only adding them and its fairly simple
            gradient = np.sum(inputs_weights)
            input_gradients[inx_row - len(rows_to_skip) //2, inx_column - len(columns_to_skip) //2] = gradient
            

      channels_combined[channel_idx] =  input_gradients
    
   
    return channels_combined     
    
        

def weight_derivative(inp: ndarray, input_pad: ndarray, input_index: map_input_weight_matrix, weights: ndarray)  -> ndarray:
    
    # Creating template for data to fill each weight will have its own deritive so the shape is the shape of weights themselfs
    weight_gradients = np.zeros(weights.shape)
    # iterating through channels
    for channel_idx, channel in enumerate(input_pad):
        # iterating through every index in weights they are shaped (channels, amount of weights per channel)
        # We pick one channel at a time using weights[channel_idx] to avoid recalculating gradients for other channels.
        #  with index = (channel, *index) we add channel info to our index
        for index in np.ndindex(weights[channel_idx].shape):
            index = (channel_idx, *index)
          
          # Get all input positions whose associated weights include the current weight index
          # this way we get every input weight was multiplied by
          # every input in input_index has described weights to it. It means those were multiplied by eachother
            input_indexes = [key for key, value in input_index[channel_idx].items()
                             if index in value]
            # if list is empty (equal to 0) we skip iteration to dont waste time and dont overwrite anythink because np.zeros are zero everywhere anyway
            if len(input_indexes) == 0:
               continue
            # we get inputs from channel based on indexes we found were used 
            # weights_inputs = [channel[row, column] for row, column in input_indexes] - old version
            rows, cols = zip(*input_indexes)
            weights_inputs = channel[rows, cols]
            
          #   Here we are just adding inputs cause our operation is basiclt weight x input (for now we dont work with dense network at all)
            gradient = np.sum(weights_inputs)
            
            
            # filling our np.zeros template
            weight_gradients[*index] = gradient
            
    
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

input_1d = np.array([[[[1,2,3,4,5],
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
                     [5,2,3,4,5]]],
                     
                     [[[1,2,3,4,5],
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
                     [5,2,3,4,5]]],

                     [[[1,2,3,4,5],
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
                     [5,2,3,4,5]]]])
# input_1d = np.array([1,2,3,4,5])
# param_1d = np.array([[1,1,1],
#                      [1,1,1],
#                      [1,1,1]])
param_1d = np.array([[1,1,1],
                     [1,1,1],
                     [1,1,1]
                     ])


   

def convolutional_input_output(input: ndarray, param: ndarray, model, flatten = True):
  output_final = np.zeros(input.shape)
  output_flatten = np.zeros(input.reshape(input.shape[0], -1).shape)
  sum_list = []
  input_grads = []
  param_grads = []

  for sample_idx, sample in enumerate(input):
                        #  conv_layer() 
    # I: Creating padded data
    #II: Creating kernels
   #III: Creating weights based on param
    conv_1 = model.conv_layer(inp = sample, param = param, jump = 0) 
    # ---------------------------------------------------------------------
                        # forward_conv()
    # I: Calculating new values by applying kernels to padded data 
    # II: Recreating kernels to make sure it matches real data              
    output, pad_input = model.forward_conv(sample, conv_1)

    # sum = model.output_sum_basic_ver(output)
    # sum_list.append(sum)
                        # backward_conv()
    # I Maping weights and inptuts relations
    #II Calculating derivarives of both inputs and params
    i_der, w_der = model.backward_conv(output, conv_1)
    input_grads.append(i_der)
    param_grads.append(w_der)
    flatten = output.flatten()
    output_flatten[sample_idx] = flatten
    output_final[sample_idx] = output


  
  param_grads = sum_param_gradients(param_grads)
  model.param_grads = param_grads
  model.input_grads = input_grads
  # input_grads = np.sum(sample for sample in input_der_list)

  return output_final, flatten, sum_list, input_grads, param_grads


   
def sum_param_gradients(sample_derivative_list: list) -> ndarray:
    grads = np.zeros((sample_derivative_list[0].shape[0], sample_derivative_list[0].shape[2], sample_derivative_list[0].shape[3]))
    for sample_idx, sample in enumerate(sample_derivative_list):
        grad = np.sum(sample, axis = 1)
        grads += grad

    return grads


   
if __name__ == "__main__":
  layer = Conv_layer()
  layer.set_layer(param = param_1d)
  flatten = layer.forward_L(input_1d)
  # print(weight_grad)
  # print(input_grad)

  # print('END')
  print(flatten)
#   print(output_final.shape)
  #   print('END')
  # print(sum_list)
#   print(input_grads)
  # print(param_grads)










 