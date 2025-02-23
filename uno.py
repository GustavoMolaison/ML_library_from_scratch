import numpy as np
from numpy import ndarray

class convenctional_model():
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
            input_pad = input_pad_calc(inp_shape, param_shape)
            
            self.kernels = get_kernels(param_shape, input_pad)
            self.kernels_weights = np.ones((self.kernels.shape[0], param.shape[0]))

        def get_info(self):
            print(f'self.kernels{self.kernels.shape}')
            print(f'self.kernels_weights{self.kernels_weights.shape}')

    def forward(self, input, conv_layer):
        return conv_ld(input, conv_layer.param, conv_layer.jump)







def _pad_ld(inp: ndarray, num: int) -> ndarray:
    # add 0 times num to begging and end of an array
     
    z = np.array([0])
    

    z = np.repeat(z, num)

    return np.concatenate([z, inp, z])


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

def kernel_forward(inp: ndarray, param: ndarray, input_pad: ndarray, jump: int = 0) -> ndarray:
    jump_calc = 0
    for o in range(inp.shape[0]):
        for p in range(param.shape[0]):
            inp[o] += param[p] * input_pad[o+p + jump_calc]
            
        jump_calc += jump    
      


    
    return inp, input_pad

def conv_ld(inp: ndarray, param: ndarray, jump: int = 0) -> ndarray:
    
    
    
    # initilization of entry data
    input_pad = input_pad_calc(inp, param, jump)
    out = np.zeros(inp.shape)
    # print(f'skipped_iteration{skipped_iteration}')
    # print(f'out.shape[0]{out.shape[0]}')
    # print(f'out{out}')
    # print(f'param_len{param_len}')
    # print(f'param_mid{param_mid}')
    # print(f'input_pad{input_pad}')
    # convulsion of one-dimension data
    jump_calc = 0
    for o in range(out.shape[0]):
        for p in range(param.shape[0]):
            out[o] += param[p] * input_pad[o+p + jump_calc]
            
        jump_calc += jump    
      


    
    return out, input_pad


def conv_ld_sum(inp: ndarray, param: ndarray) -> ndarray:

    out, input_pad = conv_ld(inp, param)
    
    return np.sum(out)

def get_kernels(param: ndarray, input_pad: ndarray) -> ndarray:
    kernels = np.zeros((input_pad.shape[0] - (param.shape[0] - 1), param.shape[0]))
    for i in range(input_pad.shape[0] - (param.shape[0] - 1)):
        print(input_pad[i : param.shape[0] + i])
        kernels[i] = input_pad[i : param.shape[0] + i]
    
    return kernels

def deriative_input(inp: ndarray, param: ndarray, input_pad: ndarray, weights: ndarray) -> ndarray:
    
    kernels = get_kernels(param, input_pad)
    input_index = {}
    # Searching for same index in a kernel
    for inx, kernel in enumerate(kernels):
        # chossing one kernel and saving its index
       
      
        # np index gets all indexes from array
        for index in np.ndindex(inp.shape):
           
            # we are gonna look for this index in our kernel
         
            # checking wether index we look for isnt to big to exist in our kernel
            if index[0] > (len(kernel) - 1) + inx:
                # print('index to big breaking')
                break
            # print(f'kernel{np.where(kernels == kernel)[0][0]}')
            # print(f'index{index}')
            # iterating over every index isnise our current kernel to compare it to input we look
            for k_value_index in np.ndindex(kernel.shape):
                # print(f'value_kernel{k_value_index}')
                # that how we calculate if the index and our kernel_value is the same  excact number at the same excact index
                if k_value_index[0] + inx == index[0]:
                   print('FOUND ONE')
                #  And saving indexs of location of our inputs inside out weights matrix
                #    print(kernel_index)
                #    print(k_value_index)
                #    print([kernel_index, *k_value_index])
                #    quit()
                   try:
                        input_index[f'input{index}'].append([inx, *k_value_index])
                        print('APPENDING')
                   except KeyError:
                        input_index[f'input{index}'] = [[inx, *k_value_index]]
    
    print(input_index)                    # print('CREATING')
    quit()
    return input_index

def input_deriative(inp: ndarray, input_index: map_input_weight_matrix, weights: ndarray) -> ndarray:
    
    input_gradients = np.zeros(inp.shape)
    for index in np.ndindex(inp.shape):
        # print(input_index)
        inputs_indexes = input_index[f'input{index}']
        print(input_index)
        quit()
        # print(f'inputs_inx{inputs_indexes}')
        inputs_weights = [weights[*i] for i in inputs_indexes]
        # print(f'{weights}\n')
        # print(inputs_weights)
        
         
        #  here gradient of kernel is one because we are only adding them and its fairly simple
        gradient = sum(inputs_weights)
        print(*index)
        quit()
        input_gradients[*index] = gradient

    return input_gradients


        


    
   
            # if index == np.array([i.index() + kernels.index(kernel) for i in kernel ]).any():

    # print(kernels)

    # for i in input_pad:

input_1d = np.array([1,2,3,4,5])
param_1d = np.array([1,1,1])

input, pad_inp = conv_ld(input_1d, param_1d)
x = map_input_weight_matrix(input, param_1d, pad_inp)
print(x)
quit()

model = convulsive_model()
conv_1 = model.conv_layer(inp = input_1d, param = param_1d, jump = 0)
output, pad_input = model.forward(input_1d, conv_1)
print(output)
# x  = conv_ld_sum(input_1d, param_1d)
# print(x)
