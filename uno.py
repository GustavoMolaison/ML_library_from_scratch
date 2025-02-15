import numpy as np
from numpy import ndarray

input_1d = np.array([1,2,3,4,5,6,7])
param_1d = np.array([1,1,1])


def _pad_ld(inp: ndarray, num: int) -> ndarray:
    # add 0 times num to begging and end of an array 
    z = np.array([0])
    z = np.repeat(z, num)
    return np.concatenate([z, inp, z])


# inp = input param = filter
def conv_ld(inp: ndarray, param: ndarray, jump: int = 1) -> ndarray:
    
    
    # filling entry data
    param_len = param.shape[0]
    param_mid = param_len // 2
    amount_of_param_passes = inp.shape[0] - (param_len - 1)
    confirmed_iteration =  (1 + ((amount_of_param_passes - 1) // (jump + 1)) )
    skipped_iteration =  amount_of_param_passes - confirmed_iteration
    input_pad = _pad_ld(inp, param_len * skipped_iteration)

    # initilization of entry data

    out = np.zeros(inp.shape)
    print(f'skipped_iteration{skipped_iteration}')
    print(f'out.shape[0]{out.shape[0]}')
    print(f'out{out}')
    print(f'param_len{param_len}')
    print(f'param_mid{param_mid}')
    print(f'input_pad{input_pad}')
    # convulsion of one-dimension data
    jump_calc = 0
    for o in range(out.shape[0]):
        for p in range(param_len):
            out[o] += param[p] * input_pad[o+p + jump_calc]
            
        jump_calc += jump    
        print(out)
        print('coool')


    
    return out

x  = conv_ld(input_1d, param_1d)
print(x)