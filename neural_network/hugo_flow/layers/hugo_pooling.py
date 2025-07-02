import numpy as np
from utils.hugo_utility import Utility as U
from numpy.lib.stride_tricks import sliding_window_view
# UNDER DEVELOPMENT
# UNDER DEVELOPMENT
# UNDER DEVELOPMENT
# UNDER DEVELOPMENT
# UNDER DEVELOPMENT
# UNDER DEVELOPMENT
# UNDER DEVELOPMENT
# UNDER DEVELOPMENT
class max_pool2d():
    def __init__(self, model=None, pool_size = (4,4)):
        self.model = model
        self.pool_size = pool_size


    def forward_L(self, input, training = True):
        print(f'input to pooled: {input.shape}')
        self.windows = sliding_window_view(input, self.pool_size, axis=(2, 3))

        self.pooled = np.max(self.windows, axis = (-2, -1))

        return self.pooled

    
    def backward_L(self, grad):
        
        pooled_expanded = self.pooled[..., np.newaxis, np.newaxis]

        grad_mask = (self.windows == pooled_expanded)

        grad_pool = grad_mask.astype(float)
        
        grad_input = grad_pool * grad[..., None, None]
        print(f'grad_input of pooled: {grad_input[0][0][0][0]}')
        quit()

        return grad_input

