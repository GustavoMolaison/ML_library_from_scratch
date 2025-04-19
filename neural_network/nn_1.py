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

def tanh(input):
    output = (np.e**input - np.e**(-input)) / (np.e**input + np.e**(-input))
    derivative = 1 - output**2
    return output, derivative

def sigmoid(input):
    output = 1/(1 + np.exp(-input))
    derivative = np.exp(-input)/ (1 + np.exp(-input))**2
    return output, derivative

def no_activation_function(input):
    return input, np.ones((input.shape))

def clipping(grads, clip_value):
    return np.clip(grads, -clip_value, clip_value)

def mse_loss(x, y):
    loss = np.mean((y - x)**2)
    loss_derivative = (2 / y.shape[0]) * (x - y)

    return loss, loss_derivative

class hugo_2_0():
    def __init__(self, loss):
        self.activation_functions = {'none': no_activation_function, 'sigmoid' : sigmoid, 'relu' : relu, 'leaky relu': leaky_relu, 'tanh': tanh}
        self.loss_methods = {'mse': mse_loss}
        self.loss = loss
        self.lr = 0.001
        self.layers = []
        pass

    def info(self):
        print(self.layers[-1].layer_weights)
    
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

        def set_layer(self, neurons_num,  input_features, activation_function, is_input_layer = False, is_output_layer = False):
            self.layer_weights = np.random.uniform(-1, 1, (input_features, neurons_num))
            # self.layer_weights = np.random.uniform(-1, 1, (neurons_num,  input_features))
            
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
            self.input = input.T
            # input = input.T
            if np.any(self.layer_weights == 0):
                self.dead_neurons = True
                print('Dead neurons appeared')
               
            # print(input.shape)
            # print(self.layer_weights.shape)
            # quit()
            self.output, self.weight_gradient  = dot_calc(input, self.layer_weights)
            self.output, self.bias_gradient  = bias_calc(self.output, self.layer_bias)
            self.output, self.af_gradient = self.layer_af_calc(self.output)
            # print(f'self.output{self.output.shape}')
            # print(f'self.af_gradient{self.af_gradient.shape}')
            # print(f'self.weight_gradient{self.weight_gradient.shape}')
         

            # self.weight_gradient = self.weight_gradient * self.af_gradient
            # self.bias_gradient = self.bias_gradient * self.af_gradient
            # quit()
            # print(f'output: {self.output}\n')
            
            # print(f'self.weight_gradient.shape{self.weight_gradient.shape}')
            # quit()

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
            # print(f'grad: {grad.shape}')
            # print(f'self.weight_gradient: {self.weight_gradient.shape}')
            # print(f'self.layer_weights: {self.layer_weights.shape}')
            grad = grad * self.af_gradient
            # quit()
            layer_weight_grad = np.dot(self.weight_gradient.T, grad)
            layer_bias_grad = np.sum(grad * self.bias_gradient, axis = 0)
            layer_input_grad = np.dot(grad, self.layer_weights.T) 
            #  gradient is how current layer affects loss, if we multiply it by weights we get how previous layer affected the loss cause input is multiplied by weights
        
            
            self.layer_weights -= layer_weight_grad * self.model.lr
            self.layer_bias -= layer_bias_grad * self.model.lr
            # print(f'new weights {self.layer_weights}')
            # print(f'new biases {self.layer_bias}\n')
            # print(f'layer_input_grad: {layer_input_grad.shape}')
            # print("W1.shape:",self.layer_weights.shape)
            # print("X.shape:", self.input.shape)
            # print("dZ1.shape:", layer_input_grad.shape)
            # print("dW1.shape (should match W1):",layer_weight_grad.shape)
            
            return layer_input_grad
    
   

    def backward(self, x, y):
        mse_loss, mse_grad = self.loss_methods[self.loss](x, y)
        # mse_loss = np.mean((y - x)**2)
        # mse_grad = (2 / y.shape[0]) * (x - y)
        print(f'grad_before: {mse_grad.shape}')
        grad = mse_grad
        grad = self.output_layer.backward_L(grad)
       
        for layer in reversed(self.layers):
            grad = layer.backward_L(grad)
            # quit()
        # quit()
        return mse_loss    
    
    def forward(self, input):
            
           output = input
           for layer in self.layers:
              output = layer.forward_L(output)
           
           if not hasattr(self, 'output_layer'):
              self.output_layer = self.Layer(self)
              print(output.shape[1])
              print(input.shape[1])
              print(output.shape)
              quit()
              self.output_layer.set_layer(input_features = output.shape[1], neurons_num = input.shape[1], activation_function = 'leaky relu')
          
           output = self.output_layer.forward_L(output)
           return output     

        

                    #  LEVEL 1 (DONE) 
# X = np.array([[0], [1], [2], [3], [4]])
# Y = np.array([[0], [1], [2], [3], [4]])
                    #  LEVEL 2 
X = np.linspace(-5, 5, 100).reshape(-1, 1)  # 100 points from -5 to 5 shape 100, 1
Y = X**2
X_training, Y_training = X[:int(X.shape[0] * 0.8)], Y[:int(Y.shape[0] * 0.8)]
X_val, Y_val = X[int(X.shape[0] * 0.8):], Y[int(Y.shape[0] * 0.8):]

# print(X.shape)
# quit()


model = hugo_2_0(loss = 'mse')

layer_I = model.Layer(model)
layer_I.set_layer(input_features= X_training.shape[1], neurons_num = 32, activation_function= 'leaky relu', is_input_layer = True)
model.add_layer(layer_I) 

dense = model.Layer(model)
dense.set_layer(input_features = 32, neurons_num = 32, activation_function= 'leaky relu', is_input_layer = False)
model.add_layer(dense, dense = 1) 

layer_0 = model.Layer(model)
layer_0.set_layer(input_features = 32, neurons_num = X_training.shape[1], activation_function= 'leaky relu', is_output_layer = True)
model.add_layer(layer_0, dense = 1) 





for i in range(100):
 print(f'EPOCH {i}')
 output = model.forward(input = X_training)
 
 loss = model.backward(output, Y_training)

 output = model.forward(input = X_val)
 val_mse_loss = np.mean((Y_val- output)**2)
#  print(f'output{output}')
 print(f'training loss: {loss}\n')
 print(f'VALIDATION loss: {val_mse_loss}\n')



# print(f'output{output}')
model.check_layers_state()
