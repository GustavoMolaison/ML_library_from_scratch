import numpy as np
from utils.hugo_utility import Utility as U
from numpy.lib.stride_tricks import sliding_window_view

class max_pool2d():
    def __init__(self, model=None, pool_size = (4,4)):
        self.model = model
        self.pool_size = pool_size


    def forward_L(self, input, training = True):

        self.windows = sliding_window_view(input, self.pool_size, axis=(2, 3))

        self.pooled = np.max(self.windows, axis = (-2, -1))
        # print(self.pooled.shape)
        # print(self.windows.shape)
        
        return self.pooled

    
    def backward_L(self, grad):
        print(self.pooled.shape)
        # print(self.windows.shape)
        pooled_expanded = self.pooled[..., np.newaxis, np.newaxis]
        # print(pooled_expanded.shape)
        

        grad_mask = (self.windows == pooled_expanded)
        # print(grad_mask.shape)
        

        grad_pool = grad_mask.astype(float)
        print(grad_pool.shape)
        print(self.windows.shape)
        # quit()
        target_array = rev_slide_loop(self.pooled, (2, 3), self.windows, (4, 5) )
       
        # print(grad_plus_windows[0][0][0][0])
        # quit()

        grad_input = grad_pool * grad[..., None, None]

        return grad_input
    
def rev_slide_loop(arrs, arr_axis,  windows, w_axis):
    windows = np.reshape(windows, (-1, windows.shape[w_axis[0]], windows.shape[w_axis[1]]))
    arrs = np.reshape(arrs, (-1, arrs.shape[arr_axis[0]], arrs.shape[arr_axis[1]]))
    print(windows.shape)
    # quit()
    print(arrs.shape)
    
    target_shape = (arrs.shape[-2] + windows.shape[-2] - 1, arrs.shape[-1] + windows.shape[-1] - 1)
    target_array = np.zeros(arrs.shape)
    target_idxs = list(np.ndindex(target_shape))
    print(target_shape)

    arrs_idx = -1
    for  win_idx, window   in enumerate(windows):

        
        print(win_idx)
        arr_idx = target_idxs[win_idx % len(target_idxs)]
         
        if arr_idx == (0,0):
            arrs_idx += 1


        if arr_idx[0] + window.shape[0] <= target_shape[0] and arr_idx[1] + window.shape[1] <= target_shape[1]: 
         
     
          print(f'arr_idx{arr_idx}')

       
          target_array[arrs_idx][arr_idx[0]: arr_idx[0] + window.shape[0],
                       arr_idx[1]: arr_idx[1] + window.shape[1]] += window
    # quit()
    # def rev_recursive_slide(windows, target_array, windows_passed):
    #   for  (win_idx, window), arr_idx in zip(enumerate(windows), np.ndindex(target_array.shape)):

    #     if arr_idx[0] + window.shape[0] > target_shape[0]:
    #       continue
    #     if arr_idx[1] + window.shape[1] > target_shape[1]:
    #       break
    #     # print(window)
    #     print(f'arr_idx{arr_idx}')

    #     # quit()
    #     target_array[arr_idx[0]: arr_idx[0] + window.shape[0], arr_idx[1]: arr_idx[1] + window.shape[1]] += window 

    #   if windows.size - win_idx - windows_passed != 0:
    #      target_array = rev_recursive_slide(windows[windows_passed :], target_array, win_idx)
      
    #   return target_array
    
    # target_array = rev_recursive_slide(windows,  target_array, 0)

    print(target_array)
    print('finish')
    quit()

test_array = np.random.rand(20, 3, 28, 28)
test = max_pool2d()
test.forward_L(input = test_array)

test.backward_L(grad= np.ones(test_array.shape))


