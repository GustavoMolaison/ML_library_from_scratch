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

def dropout(data, alpha = 0.5):
    dropout_mask = (np.random.rand(*data.shape) < alpha).astype(np.float32)
    data = data * dropout_mask
    return data

def value_f(*args):
        for v in args:
         if type(v) is not float:
            raise TypeError("Value must be a float, not {}".format(type(v).__name__))
         

def value_b(*args):
        for v in args:
         if type(v) is not bool:
            raise TypeError("Value must be a bool, not {}".format(type(v).__name__))

def value_s(*args):
        for v in args:
         if type(v) is not str:
            raise TypeError("Value must be a string, not {}".format(type(v).__name__))
      

class Dense_Layer():
        def __init__(self, model = None):
            self.model = model
            self.activation_functions = {'none': no_activation_function, 'sigmoid' : sigmoid, 'relu' : relu, 'leaky relu': leaky_relu, 'tanh': tanh}
            self.loss_methods = {'mse': mse_loss}
            # self.loss = loss
            # self.lr = lr
            self.weights_initializations = {'linear' : linear, 'he' : he, 'xavier': xavier}
            pass

        def set_layer(self, neurons_num: int, activation_function: str, weight_initialization = None, lr_update_method = 'none', input_features = None):

            self.neurons_num = neurons_num
            self.input_features = input_features

            if weight_initialization == None and self.model != None:
                self.weight_initialization = self.model.weights_initialization
            else:
                if weight_initialization == None:
                   print('You must set weight innitialization for each layer or pass model with weight innitialization setted')
                   raise LookupError
                self.weight_initialization = weight_initialization
            
            
            
            # self.layer_weights = np.random.uniform(-1, 1, (input_features, neurons_num))
            if not self.input_features  == None:
               self.layer_weights = np.random.randn(input_features,  neurons_num) * self.weights_initializations[self.weight_initialization](self.input_features, self.neurons_num)
            
            self.layer_bias = np.zeros(neurons_num,)
            self.layer_af_calc = self.activation_functions[activation_function]
            # self.layer_weights = np.array([[3]], dtype= 'float64')
            self.dead_neurons = False
            self.lr_bonus = 0
            self.lr_update_method = lr_update_method
            self.input_grads = []
            self.weights_ac_epo = []
            

    
        def forward_L(self, input):
            self.input = input.T

            if self.input_features == None:
                self.input_features = input.shape[1]
                self.layer_weights = np.random.randn(self.input_features,  self.neurons_num) * self.weights_initializations[self.weight_initialization](self.input_features, self.neurons_num)
      
            if np.any(self.layer_weights == 0):
                self.dead_neurons = True
                print('Dead neurons appeared')
               
            
            self.output, self.weight_gradient  = dot_calc(input, self.layer_weights)
            self.output, self.bias_gradient  = bias_calc(self.output, self.layer_bias)
            self.output, self.af_gradient = self.layer_af_calc(self.output)
           
            if self.model.dropout == True:
                self.output = dropout(self.output)

            return self.output
        
        # def forward_weight_gradient(self, input):
            
        #     self.weight_gradient = input
        #     return self.weight_gradient
        
        
        def backward_L(self, grad):
            # print(f'grad: {grad.shape}')
            # print(f'self.weight_gradient: {self.weight_gradient.shape}')
            # print(f'self.layer_weights: {self.layer_weights.shape}')
            grad = grad * self.af_gradient
            
            layer_weight_grad = np.dot(self.weight_gradient.T, grad) 
            layer_bias_grad = np.sum(grad * self.bias_gradient, axis = 0)   
            layer_input_grad = np.dot(grad, self.layer_weights.T) 
            
            #  gradient is how current layer affects loss, if we multiply it by weights we get how previous layer affected the loss cause input is multiplied by weights
           

            # Debuggin tools
            if hasattr(self, 'old_weights'):
                if (self.old_weights == self.layer_weights.T).all():
                    print("-----------------------\n" 
                          "WEIGHTS ARE THE SAME")
                    
            if hasattr(self, 'old_lig'):
                if (self.old_lig == layer_input_grad).all():
                    print("-----------------------\n" 
                          "INPUT GRADIENT DIDNT CHANGE")
            
            self.old_lig = layer_input_grad.copy()
            self.old_weights = self.layer_weights.T.copy()
            
            # saving data
            self.input_grads.append(layer_input_grad.copy())
            self.weights_ac_epo.append(self.layer_weights.copy())
            # print(f'{self.layer_weights}\n')
            t_w = self.layer_weights[0].copy()
            # print(layer_weight_grad * (self.model.lr + self.lr_update(layer_weight_grad)))
            # print(self.model.lr)
            # print(layer_weight_grad)
            # 
            self.layer_weights -= layer_weight_grad * (self.model.lr + self.lr_update(layer_weight_grad))
            # print(f'{self.layer_weights}\n')
            # print(f'{self.layer_weights[0]}\n')
            # print(t_w)


            # quit()

            self.layer_bias -= layer_bias_grad * (self.model.lr + self.lr_update(layer_bias_grad))
           
            
            
            return layer_input_grad
    
        def lr_update(self, grad):
            if self.lr_update_method == 'none':
                return 0
            
            if self.lr_update_method == 'Hugo_lr_bonus':
              self.lr_bonus += np.mean(np.abs(grad))  * self.model.lr**2 * 0.1
              self.lr_bonus = np.clip(self.lr_bonus, -self.model.lr, self.model.lr)
              return self.lr_bonus

    

class hugo_2_0():
    def __init__(self, loss, weight_initialization = 'none', lr = 0.001, dropout = False, dropout_alpha = 0.5):
        value_f(lr, dropout_alpha)
        value_s(loss, weight_initialization)
        value_b(dropout)
        self.activation_functions = {'none': no_activation_function, 'sigmoid' : sigmoid, 'relu' : relu, 'leaky relu': leaky_relu, 'tanh': tanh}
        self.loss_methods = {'mse': mse_loss}
        self.loss = loss
        self.lr = lr
        self.weights_initializations = {'linear' : linear, 'he' : he, 'xavier': xavier}
        self.weights_initialization = weight_initialization
        self.layers = []
        self.dropout = dropout
        self.dropout_alpha = dropout_alpha
        pass

    def info(self):
        print(self.layers[-1].layer_weights)
    
    def modify_model(self, lr = 0.001):
        self.lr =lr
    
   
       
        

    def add_layer(self, layer, dense = 1):
      
        
            for density in range(dense):
               self.layers.append(layer)
    
    def check_layers_state(self):
        dead_neuronds_indices = [i for i, layer in enumerate(self.layers) if layer.dead_neurons]
        print('Indices of layers consisiting dead neurons')
        print(dead_neuronds_indices)



    
            
            

    def backward(self, x, y):
        mse_loss, mse_grad = self.loss_methods[self.loss](x, y)
        grad = mse_grad
        grad = np.clip(grad, -1, 1)

        for layer in reversed(self.layers):
            
            grad = layer.backward_L(grad)
           
        return mse_loss    
    
    def forward(self, input):
           output = input
           for layer in self.layers:
              output = layer.forward_L(output)
           
           
           return output     
    

def run_model(model, epochs, X_training, Y_training, X_val = None, Y_val = None):
      loss_over_epochs_t = []
      loss_over_epochs_v = [] 
      for i in range(epochs):
        print(f'EPOCH {i}')
        output_t = model.forward(input = X_training)
        loss = model.backward(output_t, Y_training)
        loss_over_epochs_t.append(loss)

        # print(f'Output{output}')
        if isinstance(X_val, np.ndarray):
          output_v = model.forward(input = X_val)
          val_mse_loss = np.mean((Y_val- output_v)**2)
          loss_over_epochs_v.append(val_mse_loss)
        else:
          val_mse_loss = 0
          loss_over_epochs_v = [0]
  
        print(f'training loss: {loss}\n')
        print(f'VALIDATION loss: {val_mse_loss}\n')
      
      return loss_over_epochs_t, loss_over_epochs_v, output_t
        

               
                   #  LEVEL 7
X = np.array([
    # Cyfra "1" (klasa 0) — różne warianty
    [[0, 0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0]],
    
    [[0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0]],
    
    # Cyfra "7" (klasa 1) — różne warianty
    [[1, 1, 1, 1, 1, 1, 1],
     [0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 1, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0]],
    
    [[1, 1, 1, 1, 1, 1, 0],
     [0, 0, 0, 0, 0, 1, 0],
     [0, 0, 0, 0, 1, 0, 0],
     [0, 0, 0, 1, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0],
     [1, 0, 0, 0, 0, 0, 0]],
    
    # Litera "T" (klasa 2) — różne warianty
    [[1, 1, 1, 1, 1, 1, 1],
     [0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0]],
    
    [[1, 1, 1, 1, 1, 1, 1],
     [0, 0, 1, 0, 1, 0, 0],
     [0, 0, 1, 0, 1, 0, 0],
     [0, 0, 1, 0, 1, 0, 0],
     [0, 0, 1, 0, 1, 0, 0],
     [0, 0, 1, 0, 1, 0, 0],
     [0, 0, 1, 0, 1, 0, 0]],
    
    # Szum losowy (klasa 3) — różne warianty
    np.random.randint(0, 2, (7, 7)),
    np.random.randint(0, 2, (7, 7))
], dtype=np.float32)

# Odpowiednie etykiety:
Y = np.array([
    0, 0,   # dwa warianty "1"
    1, 1,   # dwa warianty "7"
    2, 2,   # dwa warianty "T"
    3, 3    # dwa szumy
], dtype=np.int64)
 
X = X.reshape(X.shape[0], -1)
X_training = X  
Y =  Y.reshape(-1,1)
Y_training = Y

# print(X_training.shape)
# print(Y_training.shape)
# quit()



def set_up_layers(X, Y, neurons_num, density, activation_functions: list, lr_update_method: list,  model_nn, weight_initialization: list = [None, None, None]):
        

        layer_I = Dense_Layer(model = model_nn)
        layer_I.set_layer(input_features= X.shape[1], neurons_num = neurons_num, activation_function= activation_functions[0], weight_initialization = weight_initialization[0], lr_update_method = lr_update_method[0])
        model_nn.add_layer(layer_I) 

        dense = Dense_Layer(model = model_nn)
        dense.set_layer(input_features = neurons_num, neurons_num = neurons_num, activation_function= activation_functions[1], weight_initialization = weight_initialization[1], lr_update_method = lr_update_method[1])
        model_nn.add_layer(dense, dense = density) 

        layer_0 = Dense_Layer(model = model_nn)
        layer_0.set_layer(input_features = 64, neurons_num = Y.shape[1], activation_function= activation_functions[2], weight_initialization = weight_initialization[2], lr_update_method = lr_update_method[2])
        model_nn.add_layer(layer_0, dense = 1) 

# model_uno = hugo_2_0(loss = 'mse', weight_initialization= 'linear')
# set_up_layers(X_training, Y_training, model_nn = model_uno,
#                neurons_num = 64, density = 1,
#                  activation_functions = ['leaky relu','leaky relu','sigmoid'], lr_update_method = ['Hugo_lr_bonus','Hugo_lr_bonus','Hugo_lr_bonus']
#                  weight_innitialization= [None, None, 'xavier'] )
class Hugo():
    def __init__(self, loss, weight_initialization, dropout):
        self.model = hugo_2_0(loss = loss, weight_initialization = weight_initialization, dropout = dropout)

    def set_layers(self, model_nn, X, Y, neurons_num, density, activation_functions: list, lr_update_method: list, weight_initialization: list):
        self.set_up_layers = set_up_layers(X_training, Y_training, model_nn = model_nn,
                 neurons_num = neurons_num, density = density,
                 activation_functions = activation_functions, lr_update_method = lr_update_method, 
                 weight_initialization= weight_initialization)
        
    def run(self, model_nn, epochs, X, Y):
        self.run_model = run_model(model_nn, epochs, X, Y)
        return self.run_model

if __name__ == "__main__":
     model_dos = hugo_2_0(loss = 'mse', weight_initialization= 'he', dropout = False)

     set_up_layers(X_training, Y_training, model_nn = model_dos,
                 neurons_num = 64, density = 1,
                 activation_functions = ['leaky relu','leaky relu','leaky relu'], lr_update_method = ['none','none','none'], 
                 weight_initialization= ['he', 'he', 'he'])




     epochs = 100
     # loss_over_epochs_t, loss_over_epochs_v,output = run_model(model, epochs, X_training, Y_training, X_val, Y_val)

     # print('                A')
     # print(f'training loss: {loss_over_epochs_t[-1]}')
     # print(f'VALIDATION loss: {loss_over_epochs_v[-1]}\n')
     X_training = min_max_normalize(X_training)
     loss_over_epochs_t2, loss_over_epochs_v2, output = run_model(model_dos, epochs, X_training, Y_training)
     # loss_over_epochs_t2, loss_over_epochs_v2, output = run_model(model_dos, epochs, X_training, Y_training, X_val, Y_val)
     print(f'output before rounding {output}')
     output = np.round(output).astype(int)
     accuracy = np.mean(output == Y_training)
     print(f' output after rounding\n{output}')
     print('\n')
     print(f'True data \n{Y_training}')

     print('                B')
     print(f'training loss: {loss_over_epochs_t2[-1]}')
     print(f'training accuracy: {accuracy}')
     print(f'VALIDATION loss: {loss_over_epochs_v2[-1]}\n')
     model_dos.check_layers_state()


     plt.figure(figsize=(12, 6))
     epochs = np.arange(1, epochs + 1)


# Training Loss
# plt.plot(epochs, loss_over_epochs_t, label='Train Loss - Net A', linestyle='--', color='blue')
# plt.plot(epochs, loss_over_epochs_t2, label='Train Loss - Net B', linestyle='--', color='green')

# # Validation Loss
# # plt.plot(epochs, loss_over_epochs_v, label='Val Loss - Net A', linestyle='-', color='blue')
# plt.plot(epochs, loss_over_epochs_v2, label='Val Loss - Net B', linestyle='-', color='green')

# # Labels and Title
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.title('Training vs Validation Loss for Neural Networks')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()

# plt.show()