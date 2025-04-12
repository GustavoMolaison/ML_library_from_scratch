import numpy as np
import pandas as pd


class hugo_2_0():
    def __init__(self):
        
        pass

    def initilialization(self, output_shape):
        
        self.output_shape = output_shape
        

    

    class Layer():
        def __init__(self):
            
            pass

        def set_layer(self, neurons_num, input_shape):
            self.layer = np.random.uniform(0, 5, (input_shape[1], neurons_num))
            # print(input_shape)
            print(self.layer)
            # quit()

    
        def forward_L(self, input):
            output = np.dot(input, self.layer)
            # print(output)
            return output

        def backward_L(self, x, y):
            mse_loss = np.mean((y - x)**2)
            mse_grad = (2 / y.shape[0]) * (y - x)
            print(f'mse_loss: {mse_loss}')
            print(f'mse_grad: {mse_grad}\n')

            print(f'old weights {self.layer}')
            self.layer = np.repeat(self.layer, mse_grad.shape[0], axis = 0)
            # print(self.layer)
            # quit()
            self.layer += mse_grad
            print(f'new weights {self.layer}')

            # MSE GRAD POKAZUJE JA KZMIENI SIE MSE GDY ZWIEKSZYMY OUTPUT O JEDEN



X = np.array([[0], [1], [2], [3], [4]])
Y = np.array([[0], [1], [2], [3], [4]])
# print(X.shape)
# quit()


model = hugo_2_0()
layer_0 = model.Layer
layer_0.set_layer(model, input_shape= X.shape, neurons_num = 1) 
output = layer_0.forward_L(model, input = X)
layer_0.backward_L(model, output, Y)