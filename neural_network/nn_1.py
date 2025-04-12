import numpy as np
import pandas as pd

def dot_calc(x, w):
    output = np.dot(x, w)
    deriative = x
    return output, deriative

def bias_calc(x, b):
    output = x + b
    deriative = 1
    return output, deriative

def sigmoid(input):
    output = 1/(1 + np.exp(-input))
    deriative = np.exp(-input)/ (1 + np.exp(-input))**2
    return output, deriative
def no_activation_function(input):
    return input, 1


class hugo_2_0():
    def __init__(self):
        self.activation_functions = {'none': no_activation_function, 'sigmoid' : sigmoid}
        pass

    def initilialization(self, output_shape):
        
        self.output_shape = output_shape
        

    

    class Layer():
        def __init__(self):
            
            pass

        def set_layer(self, neurons_num, input_shape, activation_function, is_input_layer = False):
            self.layer_weights = np.random.uniform(-1, 1, (input_shape[1], neurons_num))
            self.layer_bias = 0
            self.layer_af_calc = self.activation_functions[activation_function]
            # self.layer_weights = np.array([[3]], dtype= 'float64')
            self.input_layer = is_input_layer
            
            # quit()
            # print(self.layer_weights)
            # quit()

    
        def forward_L(self, input):
            self.input = input
            # print(self.layer_weights.shape)
            # print(input.shape)
            self.output, self.weight_gradient  = dot_calc(input, self.layer_weights)
            self.output, self.bias_gradient  = bias_calc(self.output, self.layer_bias)
            self.output, self.af_gradient = self.layer_af_calc(self.output)
            # print(self.af_gradient)
            # quit()

            self.weight_gradient = self.weight_gradient * self.af_gradient
            self.bias_gradient = self.bias_gradient * self.af_gradient
            # quit()
            # print(f'output: {self.output}\n')
            
            

            # self.weight_gradient = input
            # self.bias_gradient = 1
            # print(f'self.input \n {self.input}')
            # print(f'self.layer_weights \n {self.layer_weights}')
            # print(f'self.output \n {self.output}')
            # print('\n')
            return self.output
        
        # def forward_weight_gradient(self, input):
            
        #     self.weight_gradient = input
        #     return self.weight_gradient

        def backward_L(self, x, y):
            mse_loss = np.mean((y - x)**2)
            mse_grad = (2 / y.shape[0]) * (x - y)
           
           
            layer_weight_grad = np.sum(mse_grad * self.weight_gradient)
            layer_bias_grad = np.sum(mse_grad * self.bias_gradient)
            
            print(f'mse_loss: {mse_loss}')
            # print(f'mse_grad: {mse_grad}\n')
            # print(f'layer_weight_grad: {layer_weight_grad}\n')
            # print(f'layer_weights: {self.layer_weights}\n')
            # print(f'old weights {self.layer_weights}')
            # self.layer_weights = np.repeat(self.layer_weights, mse_grad.shape[0], axis = 0)
            # print(self.layer_weights)
            # quit()
            self.layer_weights -= layer_weight_grad * 0.01
            self.layer_bias -= layer_bias_grad * 0.1
            print(f'new weights {self.layer_weights}')
            print(f'new biases {self.layer_bias}\n')



                    #  LEVEL 1 (DONE) 
# X = np.array([[0], [1], [2], [3], [4]])
# Y = np.array([[0], [1], [2], [3], [4]])
                    #  LEVEL 2 
X = np.linspace(-5, 5, 100).reshape(-1, 1)  # 100 points from -5 to 5
Y = X**2
# print(X.shape)
# quit()


model = hugo_2_0()
layer_0 = model.Layer
layer_0.set_layer(model, input_shape= X.shape, neurons_num = 1, activation_function= 'none') 


for i in range(600):
 print(f'EPOCH {i}\n')
 output = layer_0.forward_L(model, input = X)
 layer_0.backward_L(model, output, Y)

X = np.linspace(-5, 5, 100).reshape(-1, 1)  # 100 points from -5 to 5
Y = X**2

