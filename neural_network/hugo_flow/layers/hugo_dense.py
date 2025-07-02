from ..utils.hugo_utility import Utility as U
import numpy as np






class Dense_Layer():
        def __init__(self, model = None):
            self.model = model
            self.activation_functions = {'none': U.no_activation_function, 'sigmoid' : U.sigmoid, 'relu' : U.relu, 'leaky relu': U.leaky_relu, 'tanh': U.tanh}
            self.loss_methods = {'mse': U.mse_loss, 'cross_entropy': U.cross_entropy_loss}
            self.update_methods = {'gradient descent': U.basic_grad_update, 'SGD': U.SGD_momentum}
            # self.loss = loss
            # self.lr = lr
            self.weights_initializations = {'linear' : U.linear, 'he' : U.he, 'xavier': U.xavier}
            
            pass

        def set_layer(self, neurons_num: int, activation_function: str, weight_initialization = None, update_method = 'gradient descent', input_features = None):

            self.neurons_num = neurons_num
            self.input_features = input_features
            self.activation_function = activation_function
            # this part of code sets parameters using outer model class if user didnt set them individually for layer
            if weight_initialization == None and self.model != None:
                self.weight_initialization = self.model.weights_initialization
            else:
                if weight_initialization == None:
                   print('You must set weight innitialization for each layer or pass model with weight innitialization setted')
                   raise LookupError
                self.weight_initialization = weight_initialization
            
            if update_method == 'gradient descent' and self.model !='gradient descent':
                self.update_method = self.model.update_method
            else:
                self.update_method = update_method
            
            
            # self.layer_weights = np.random.uniform(-1, 1, (input_features, neurons_num))
            if not self.input_features  == None:
               self.layer_weights = np.random.randn(input_features,  neurons_num) * self.weights_initializations[self.weight_initialization](self.input_features, self.neurons_num)
            
            self.layer_bias = np.zeros(neurons_num,)
            self.layer_af_calc = self.activation_functions[activation_function]
            # self.layer_weights = np.array([[3]], dtype= 'float64')
            self.dead_neurons = False
            self.lr_bonus = 0
            self.velocity_w = 0
            self.velocity_b = 0
            self.input_grads = []
            self.weights_ac_epo = []
            

    
        def forward_L(self, input, training):
            self.input = input.T

            if self.input_features == None:
                self.input_features = input.shape[1]
                self.layer_weights = np.random.randn(self.input_features,  self.neurons_num) * self.weights_initializations[self.weight_initialization](self.input_features, self.neurons_num)
      
            if np.any(self.layer_weights == 0):
                self.dead_neurons = True
                print('Dead neurons appeared')
               
            if training == True:
              self.output, self.weight_gradient  = U.dot_calc(input, self.layer_weights)
              self.output, self.bias_gradient  = U.bias_calc(self.output, self.layer_bias)
              self.output, self.af_gradient = self.layer_af_calc(self.output)
           
            if self.model.dropout == True:
                self.output = U.dropout(self.output)

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
            # if hasattr(self, 'old_weights'):
            #     if (self.old_weights == self.layer_weights.T).all():
            #         print("-----------------------\n" 
            #               "WEIGHTS ARE THE SAME")
                    
            # if hasattr(self, 'old_lig'):
            #     if (self.old_lig == layer_input_grad).all():
            #         print("-----------------------\n" 
            #               "INPUT GRADIENT DIDNT CHANGE")
            
            # self.old_lig = layer_input_grad.copy()
            # self.old_weights = self.layer_weights.T.copy()
            
            # saving data
            # self.input_grads.append(layer_input_grad.copy())
            # self.weights_ac_epo.append(self.layer_weights.copy())
            # print(f'{self.layer_weights}\n')
            # t_w = self.layer_weights[0].copy()
            # print(layer_weight_grad * (self.model.lr + self.lr_update(layer_weight_grad)))
            # print(self.model.lr)
            # print(layer_weight_grad)
            # velocity is gradient just diffrent name
            # here is chossen momentum optimizers being used  
            # lr have to be frist parameter and grad second cause of args basic_grad_update function remember!
            self.velocity_w = self.update_methods[self.update_method](self.model.lr,  layer_weight_grad,  self.velocity_w)
            self.velocity_b = self.update_methods[self.update_method](self.model.lr,  layer_bias_grad,    self.velocity_b)

            self.layer_weights -= self.velocity_w
            self.layer_bias -= self.velocity_b
            # self.layer_weights -= layer_weight_grad * self.model.lr * self.lr_update_methods
          
          
            # self.layer_bias -= layer_bias_grad * self.model.lr  
            
            
            return layer_input_grad