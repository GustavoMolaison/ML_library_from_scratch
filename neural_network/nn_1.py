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
        self.layers = []
        pass

    def info(self):
        
        print(self.layers[-1].layer_weights.shape)
       
        

    def add_layer(self, layer, dense = 1):
      
        if layer.is_output_layer == True:
           self.output_layer = layer
        
        else:   
            for density in range(dense):
               self.layers.append(layer)
       


    class Layer():
        def __init__(self, model):
            self.model = model
            pass

        def set_layer(self, neurons_num, batch_size, activation_function, is_input_layer = False, is_output_layer = False):
            self.layer_weights = np.random.uniform(-1, 1, (batch_size, neurons_num))
            
            self.layer_bias = 0
            self.layer_af_calc = self.model.activation_functions[activation_function]
            # self.layer_weights = np.array([[3]], dtype= 'float64')
            self.is_input_layer = is_input_layer
            self.is_output_layer = is_output_layer
            # quit()
            # print(self.layer_weights)
            # quit()

    
        def forward_L(self, input):
            self.input = input
            print(input.shape)
            print(self.layer_weights.shape)
            # quit()
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

        def backward_L(self, grad):
           
            layer_weight_grad = np.sum(grad * self.weight_gradient)
            layer_bias_grad = np.sum(grad * self.bias_gradient)
            layer_input_grad = np.sum(grad * self.input)
            
            self.layer_weights -= layer_weight_grad * 0.01
            self.layer_bias -= layer_bias_grad * 0.1
            print(f'new weights {self.layer_weights}')
            print(f'new biases {self.layer_bias}\n')

            return layer_input_grad
    
    def forward(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward_L(output)

        if not hasattr(self, 'output_layer'):
          self.output_layer = self.Layer(self)
          self.output_layer.set_layer(batch_size = output.shape[1], neurons_num = input.shape[1], activation_function = 'none')
        
        output = self.output_layer.forward_L(output)

        return output

   

    def backward(self, x, y):
        mse_loss = np.mean((y - x)**2)
        mse_grad = (2 / y.shape[0]) * (x - y)
        print(f'mse_loss{mse_loss}')
        print(f'mse_grad{mse_grad}')
        grad = mse_grad
        for layer in self.layers.reverse():
            grad = layer.backward_L(grad)

        

                    #  LEVEL 1 (DONE) 
# X = np.array([[0], [1], [2], [3], [4]])
# Y = np.array([[0], [1], [2], [3], [4]])
                    #  LEVEL 2 
X = np.linspace(-5, 5, 100).reshape(-1, 1)  # 100 points from -5 to 5 shape 100, 1
Y = X**2
# print(X.shape)
# quit()


model = hugo_2_0()
layer_0 = model.Layer(model)
layer_0.set_layer(batch_size= X.shape[1], neurons_num = 5, activation_function= 'none', is_input_layer = True)
model.add_layer(layer_0) 
dense = model.Layer(model)
dense.set_layer(batch_size = 5, neurons_num = 5, activation_function= 'none', is_input_layer = False)
model.add_layer(dense, dense = 5) 
model.info()



for i in range(100):
 print(f'EPOCH {i}\n')
 output = model.forward(input = X)
 print(f'output{output}')
#  quit()
 model.backward(output, Y)

X = np.linspace(-5, 5, 100).reshape(-1, 1)  # 100 points from -5 to 5
Y = X**2

