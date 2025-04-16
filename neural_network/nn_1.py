import numpy as np
import pandas as pd

def dot_calc(x, w):
    output = np.dot(x, w)
    derivative = x
    return output, derivative

def bias_calc(x, b):
    output = x + b
    derivative = 1
    return output, derivative

def relu(input):
    output = np.maximum(0, input)
    derivative = (output > 0).astype(float) # astype change bool True to 1 and False to 0 our deriatives
    return output, derivative

def leaky_relu(input, alpha = 0.01):
    output = np.where(input < 0, input * alpha, input)
    derivative = np.where(output < input, alpha, 1.0)
    return output, derivative

def sigmoid(input):
    output = 1/(1 + np.exp(-input))
    derivative = np.exp(-input)/ (1 + np.exp(-input))**2
    return output, derivative

def no_activation_function(input):
    return input, 1

def clipping(grads, clip_value):
    return np.clip(grads, -clip_value, clip_value)


class hugo_2_0():
    def __init__(self):
        self.activation_functions = {'none': no_activation_function, 'sigmoid' : sigmoid, 'relu' : relu, 'leaky relu': leaky_relu}
        self.lr = 0.01
        self.layers = []
        pass

    def info(self):
        print(self.layers[-1].layer_weights.shape)
    
    def modify_model(self, lr = 0.01):
        self.lr =lr
        
       
        

    def add_layer(self, layer, dense = 1):
      
        if layer.is_output_layer == True:
           self.output_layer = layer
        
        else:   
            for density in range(dense):
               self.layers.append(layer)
    
    def check_layers_state(self):
        dead_neuronds_indices = [i for i, layer in enumerate(self.layers) if layer.dead_neurons]
        print('Indices of layers consisiting dead neurons')
        print(dead_neuronds_indices)



    class Layer():
        def __init__(self, model):
            self.model = model
            pass

        def set_layer(self, neurons_num, batch_size, activation_function, is_input_layer = False, is_output_layer = False):
            self.layer_weights = np.random.uniform(-1, 1, (batch_size, neurons_num))
            
            self.layer_bias = np.zeros(neurons_num,)
            self.layer_af_calc = self.model.activation_functions[activation_function]
            # self.layer_weights = np.array([[3]], dtype= 'float64')
            self.is_input_layer = is_input_layer
            self.is_output_layer = is_output_layer
            self.dead_neurons = False
            # quit()
            # print(self.layer_weights)
            # quit()

    
        def forward_L(self, input):
            self.input = input
            if np.any(self.layer_weights == 0):
                self.dead_neurons = True
                print('Dead neurons appeared')
               
            # print(input.shape)
            # print(self.layer_weights.shape)
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
            # print(f'grad: {grad}')
            
            layer_weight_grad = np.dot(self.weight_gradient.T, grad)
            layer_bias_grad = np.sum(grad * self.bias_gradient, axis = 0)
            print(f'grad: {grad}')
            print(f'self.layer_bias: {self.layer_bias}')
            print(f'shape self.bias_gradient: {layer_bias_grad}')
            # quit()
            layer_input_grad = np.dot(grad, self.layer_weights.T) 
            #  gradient is how current layer affects loss, if we multiply it by weights we get how previous layer affected the loss cause input is multiplied by weights
        
            
            self.layer_weights -= layer_weight_grad * self.model.lr
            self.layer_bias -= layer_bias_grad * self.model.lr
            # print(f'new weights {self.layer_weights}')
            # print(f'new biases {self.layer_bias}\n')

            return layer_input_grad
    
   

    def backward(self, x, y):
        mse_loss = np.mean((y - x)**2)
        mse_grad = (2 / y.shape[0]) * (x - y)
     
        grad = mse_grad
        grad = self.output_layer.backward_L(grad)
       
        for layer in reversed(self.layers):
            grad = layer.backward_L(grad)
            
        return mse_loss    
    
    def forward(self, input):
           output = input
           for layer in self.layers:
              output = layer.forward_L(output)

           if not hasattr(self, 'output_layer'):
              self.output_layer = self.Layer(self)
              self.output_layer.set_layer(batch_size = output.shape[1], neurons_num = input.shape[1], activation_function = 'none')
          
           output = self.output_layer.forward_L(output)
           return output     

        

                    #  LEVEL 1 (DONE) 
# X = np.array([[0], [1], [2], [3], [4]])
# Y = np.array([[0], [1], [2], [3], [4]])
                    #  LEVEL 2 
X = np.linspace(-5, 5, 100).reshape(-1, 1)  # 100 points from -5 to 5 shape 100, 1
Y = X**2
# print(X.shape)
# quit()


model = hugo_2_0()

layer_I = model.Layer(model)
layer_I.set_layer(batch_size= X.shape[1], neurons_num = 5, activation_function= 'none', is_input_layer = True)
model.add_layer(layer_I) 

dense = model.Layer(model)
dense.set_layer(batch_size = 5, neurons_num = 5, activation_function= 'leaky relu', is_input_layer = False)
model.add_layer(dense, dense = 1) 





for i in range(9):
 print(f'EPOCH {i}')
 output = model.forward(input = X)
 
 loss = model.backward(output, Y)
#  print(f'output{output}')
 print(f'loss: {loss}\n')

# print(f'output{output}')
model.check_layers_state()
