import numpy as np
from utils.hugo_utility import Utility as U 
np.set_printoptions(threshold=np.inf)


class Hugo_Agent():
    def __init__(self, loss, mode, weight_initialization = 'none', clip_method ='norm clipping', update_method = 'gradient descent', lr = 0.001, dropout = False, dropout_alpha = 0.5, max_grad = 1):
        
        # Agent type (Policy Network,  Value Network, Q-Network)
        self.mode = mode
        
        
        
        
        U.value_f(lr, dropout_alpha)
        U.value_s(loss, weight_initialization)
        U.value_b(dropout)
        self.activation_functions = {'none': U.no_activation_function, 'sigmoid' : U.sigmoid, 'relu' : U.relu, 'leaky relu': U.leaky_relu, 'tanh': U.tanh}
        self.loss_methods = {'policy network': U.Agent.policy_grad}
        self.clipping_methods = {'norm clipping': U.norm_clipping}
        self.loss = loss
        self.lr = lr
        self.weights_initializations = {'linear' : U.linear, 'he' : U.he, 'xavier': U.xavier}
        self.weights_initialization = weight_initialization
        self.layers = []
        self.dropout = dropout
        self.dropout_alpha = dropout_alpha
        self.update_method = update_method
        self.clip_method = clip_method
        self.max_grad = max_grad
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



    
            
            

    def backward(self, x, action, reward, training = True):
       
        loss, grad = self.loss_methods[self.mode](x, action, reward)
        if training == False:
            return loss
        grad = grad
        grad = self.clipping_methods[self.clip_method](grad, self.max_grad)
        

        for layer in reversed(self.layers):
            
            grad = layer.backward_L(grad)
            grad = self.clipping_methods[self.clip_method](grad, self.max_grad)
           
        return loss    
    
    def forward(self, input, training):
           output = input
           import time
           for layer in self.layers:
            #   print('Layer Done!')
              # start = time.perf_counter()
              output = layer.forward_L(output, training)
              # end = time.perf_counter()
            #   print(f"Execution time {layer}: {end - start:.4f} seconds")
           
           
           return output     
    

def run_model(model, epochs, X_training, Y_training, X_val = None, Y_val = None, training = True):
      loss_over_epochs_t = []
      loss_over_epochs_v = [] 
      import time
      for i in range(epochs):
        start = time.perf_counter()
        print(f'EPOCH {i}')
        output_t = model.forward(input = X_training, training = True)
        loss_t = model.backward(output_t, Y_training, training = True)
        loss_over_epochs_t.append(loss_t)

        # print(f'Output{output}')
        if isinstance(X_val, np.ndarray):
          output_v = model.forward(input = X_val, training = False)
          loss_v = model.backward(output_v, Y_val, training = False)
          loss_over_epochs_v.append(loss_v)
        else:
          loss_v = 0
          loss_over_epochs_v = [0]
  
        print(f'training loss: {loss_t}\n')
        print(f'VALIDATION loss: {loss_v}\n')
        end = time.perf_counter()
        print(f"Execution time of epoch: {end - start:.4f} seconds")
      if isinstance(X_val, np.ndarray):
        return loss_over_epochs_t, loss_over_epochs_v, output_t, output_v
      else:   
        return loss_over_epochs_t, output_t
               
 

# SET UP LAYER FEATURE NEED CHANGE OUTDATED
def set_up_layers(X, Y, neurons_num, density, activation_functions: list, lr_update_method: list,  model_nn, weight_initialization: list = [None, None, None]):
        
        from layers import hugo_dense as dense

        layer_I = dense(model = model_nn)
        layer_I.set_layer(input_features= X.shape[1], neurons_num = neurons_num, activation_function= activation_functions[0], weight_initialization = weight_initialization[0], update_method = lr_update_method[0])
        model_nn.add_layer(layer_I) 

        dense = dense(model = model_nn)
        dense.set_layer(input_features = neurons_num, neurons_num = neurons_num, activation_function= activation_functions[1], weight_initialization = weight_initialization[1], update_method = lr_update_method[1])
        model_nn.add_layer(dense, dense = density) 

        layer_0 = dense(model = model_nn)
        layer_0.set_layer(input_features = 64, neurons_num = Y.shape[1], activation_function= activation_functions[2], weight_initialization = weight_initialization[2], update_method = lr_update_method[2])
        model_nn.add_layer(layer_0, dense = 1) 


class Hugo():
    def __init__(self, loss, weight_initialization, dropout, lr, clip_method = 'norm clipping', update_method = 'gradient descent', max_grad = 1):
        self.model = hugo_2_0(loss = loss, update_method= update_method, weight_initialization = weight_initialization, dropout = dropout, lr = lr, max_grad = max_grad)

    
        
    def run(self, model_nn, epochs, X, Y, X_val = None, Y_val = None):
        self.run_model = run_model(model_nn, epochs, X, Y, X_val = X_val, Y_val = Y_val)
        return self.run_model
















# if __name__ == "__main__":
    