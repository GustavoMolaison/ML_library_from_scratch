import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def linear(*args):
    return 1

def he(*args):
    return np.sqrt(2. / (args[1]))

def xavier(*args):
    limit = np.sqrt(6 / (args[0] + args[1]))
    return np.random.uniform(-limit, limit, size=(args[0], args[1]))

def min_max_normalize(data):
    return (data - np.min(data, axis = 0)) / (np.max(data, axis = 0) - np.min(data, axis=0) + 1e-8)


    

class hugo_2_0():
    def __init__(self, loss, weight_initialization = 'none', lr = 0.001):
        self.activation_functions = {'none': no_activation_function, 'sigmoid' : sigmoid, 'relu' : relu, 'leaky relu': leaky_relu, 'tanh': tanh}
        self.loss_methods = {'mse': mse_loss}
        self.loss = loss
        self.lr = lr
        self.weights_initializations = {'linear' : linear, 'he' : he, 'xavier': xavier}
        self.weights_initialization = weight_initialization
        self.layers = []
        pass

    def info(self):
        print(self.layers[-1].layer_weights)
    
    def modify_model(self, lr = 0.001):
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

        def set_layer(self, neurons_num,  input_features, activation_function, is_input_layer = False, is_output_layer = False, lr_update_method = 'none', weight_initilization = 'notchossed' ):
            if weight_initilization == 'notchossed':
                self.weight_initilization = self.model.weights_initialization
            else:
                self.weight_initilization = weight_initilization
            print(neurons_num)
          
            # self.layer_weights = np.random.uniform(-1, 1, (input_features, neurons_num))
            self.layer_weights = np.random.randn(input_features,  neurons_num) * self.model.weights_initializations[self.weight_initilization](input_features, neurons_num)
            
            self.layer_bias = np.zeros(neurons_num,)
            self.layer_af_calc = self.model.activation_functions[activation_function]
            # self.layer_weights = np.array([[3]], dtype= 'float64')
            self.is_input_layer = is_input_layer
            self.is_output_layer = is_output_layer
            self.dead_neurons = False
            self.lr_bonus = 0
            self.lr_update_method = lr_update_method
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
            
        
            self.layer_weights -= layer_weight_grad * (self.model.lr + self.lr_update(layer_weight_grad))
            self.layer_bias -= layer_bias_grad * (self.model.lr + self.lr_update(layer_bias_grad))
            
            
            return layer_input_grad
    
        def lr_update(self, grad):
            if self.lr_update_method == 'none':
                return 0
            
            if self.lr_update_method == 'Hugo_lr_bonus':
              self.lr_bonus += np.mean(np.abs(grad))  * self.model.lr**2 * 0.1
              self.lr_bonus = np.clip(self.lr_bonus, -self.model.lr, self.model.lr)
              return self.lr_bonus
        
            
            

    def backward(self, x, y):
        mse_loss, mse_grad = self.loss_methods[self.loss](x, y)
        # mse_loss = np.mean((y - x)**2)
        # mse_grad = (2 / y.shape[0]) * (x - y)
        grad = mse_grad
        grad = self.output_layer.backward_L(grad)
        grad = np.clip(grad, -1, 1)
       
        for layer in reversed(self.layers):
            grad = layer.backward_L(grad)
            # quit()
        # quit()
        return mse_loss    
    
    def forward(self, input):
           input = min_max_normalize(input)
           output = input
           for layer in self.layers:
              output = layer.forward_L(output)
           
           if not hasattr(self, 'output_layer'):
              self.output_layer = self.Layer(self)
              
              self.output_layer.set_layer(input_features = output.shape[1], neurons_num = input.shape[1], activation_function = 'none')
          
           output = self.output_layer.forward_L(output)
           return output     
    

def run_model(model, epochs, X_training, Y_training, X_val, Y_val):
      loss_over_epochs_t = []
      loss_over_epochs_v = [] 
      for i in range(epochs):
        print(f'EPOCH {i}')
        output_t = model.forward(input = X_training)
        loss = model.backward(output_t, Y_training)
        loss_over_epochs_t.append(loss)

        # print(f'Output{output}')
        
        output_v = model.forward(input = X_val)
        val_mse_loss = np.mean((Y_val- output_v)**2)
        loss_over_epochs_v.append(val_mse_loss)
  
        print(f'training loss: {loss}\n')
        print(f'VALIDATION loss: {val_mse_loss}\n')
      
      return loss_over_epochs_t, loss_over_epochs_v, output_t
        

                    #  LEVEL 1 (DONE) 
# X = np.array([[0], [1], [2], [3], [4]])
# Y = np.array([[0], [1], [2], [3], [4]])

                    #  LEVEL 2 
# X = np.linspace(-5, 5, 100).reshape(-1, 1)  # 100 points from -5 to 5 shape 100, 1
# Y = min_max_normalize(X**2)

                    #  LEVEL 3
# X = np.linspace(-10, 10, 500).reshape(-1, 1)
# X = np.hstack([X, np.sin(X)])
# Y = min_max_normalize(np.sin(X[:, 0]) + 0.3 * np.cos(3 * X[:, 0]) + 0.1 * X[:, 0]**2)
# Y = Y.reshape(-1, 1)
                    #  LEVEL 4
# X = np.linspace(-10, 10, 500).reshape(-1, 1)
# Y = np.piecewise(X.flatten(),
#                  [X.flatten() < 0, X.flatten() >= 0],
#                  [lambda x: np.sin(x) + np.random.normal(0, 0.1, x.shape),
#                   lambda x: np.log1p(x) + np.random.normal(0, 0.1, x.shape)])
# Y = min_max_normalize(Y.reshape(-1, 1))
                    #  LEVEL 5

X = np.linspace(-5, 5, 1000).reshape(-1, 1)
Y = np.sin(5 * X) * np.cos(2 * X) + 0.1 * X

Y = min_max_normalize(Y.reshape(-1, 1))

X_training, Y_training = X[:int(X.shape[0] * 0.8)], Y[:int(Y.shape[0] * 0.8)]
X_val, Y_val = X[int(X.shape[0] * 0.8):], Y[int(Y.shape[0] * 0.8):]


# quit()

# print(X.shape)
# quit()


model = hugo_2_0(loss = 'mse', weight_initialization= 'linear')

layer_I = model.Layer(model)
layer_I.set_layer(input_features= X_training.shape[1], neurons_num = 64, activation_function= 'leaky relu', is_input_layer = True, lr_update_method = 'Hugo_lr_bonus')
model.add_layer(layer_I) 

dense = model.Layer(model)
dense.set_layer(input_features = 64, neurons_num = 64, activation_function= 'leaky relu', is_input_layer = False, lr_update_method = 'Hugo_lr_bonus')
model.add_layer(dense, dense = 1) 

layer_0 = model.Layer(model)
layer_0.set_layer(input_features = 64, neurons_num = X_training.shape[1], activation_function= 'leaky relu', is_output_layer = True, lr_update_method = 'Hugo_lr_bonus')
model.add_layer(layer_0, dense = 1) 


model2 = hugo_2_0(loss = 'mse', weight_initialization= 'he')

layer_I = model2.Layer(model2)
layer_I.set_layer(input_features= X_training.shape[1], neurons_num = 32, activation_function= 'leaky relu', is_input_layer = True, lr_update_method ='none')
model2.add_layer(layer_I) 

dense = model2.Layer(model2)
dense.set_layer(input_features = 32, neurons_num = 32, activation_function= 'leaky relu', is_input_layer = False, lr_update_method = 'none')
model2.add_layer(dense, dense = 10) 

layer_0 = model2.Layer(model2)
layer_0.set_layer(input_features = 32, neurons_num = Y_training.shape[1], activation_function= 'leaky relu', is_output_layer = True, lr_update_method = 'none',
                  weight_initilization= 'xavier')
model2.add_layer(layer_0, dense = 1) 



epochs = 100
# loss_over_epochs_t, loss_over_epochs_v,output = run_model(model, epochs, X_training, Y_training, X_val, Y_val)
# print('                A')
# print(f'training loss: {loss_over_epochs_t[-1]}')
# print(f'VALIDATION loss: {loss_over_epochs_v[-1]}\n')

loss_over_epochs_t2, loss_over_epochs_v2, output = run_model(model2, epochs, X_training, Y_training, X_val, Y_val)
print('                B')
print(f'training loss: {loss_over_epochs_t2[-1]}')
print(f'VALIDATION loss: {loss_over_epochs_v2[-1]}\n')
model.check_layers_state()


plt.figure(figsize=(12, 6))
epochs = np.arange(1, epochs + 1)


# Training Loss
# plt.plot(epochs, loss_over_epochs_t, label='Train Loss - Net A', linestyle='--', color='blue')
plt.plot(epochs, loss_over_epochs_t2, label='Train Loss - Net B', linestyle='--', color='green')

# Validation Loss
# plt.plot(epochs, loss_over_epochs_v, label='Val Loss - Net A', linestyle='-', color='blue')
plt.plot(epochs, loss_over_epochs_v2, label='Val Loss - Net B', linestyle='-', color='green')

# Labels and Title
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss for Neural Networks')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()