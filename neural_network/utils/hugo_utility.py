import numpy as np
from numpy import ndarray
np.set_printoptions(threshold=np.inf)
# This file consinst all usefull function for deep learing its practical to take them from one place
# Many could be taken from numpy but writing them by myself with a litte help of chat gpt helps me understand them 
class Utility():
    def __init__(self):
        pass

    def dot_calc(x, w):
        output = np.dot(x, w)
        derivative = x
        return output, derivative

    def bias_calc(x, b):
        output = x + b
        derivative = 1
        return output, derivative
    
    def basic_grad_update(*args):
        return args[0] * args[1]
    
    def SGD_momentum(lr, grad, prev_velocity):
        return 0.9 * prev_velocity + lr * grad
        
    def norm_clipping(grad, limit = 1):
        norm = np.sqrt(np.sum(grad ** 2))
        if norm > limit:
            grad = grad * (limit/norm)
        return grad
        
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
    
    def softmax(z):
       e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
       return e_z / np.sum(e_z, axis=1, keepdims=True)
    
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
    
    def cross_entropy_loss(x, y, eps=1e-15):
    
       # Compute softmax probabilities
       
       probs = Utility.softmax(x)
       
       # Cross-entropy loss
       loss = -np.sum(y * np.log(probs +eps), axis=1)

       # Gradient: Case 1
       loss_derivative = probs - y
       loss = np.mean(loss)

       return loss, loss_derivative
    
    def one_hot_encoding(y, num_classes):
        one_hot = np.zeros((y.size, num_classes))
        one_hot[np.arange(y.size), y] = 1
        return one_hot
    
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
      
    def channels_pad_to_shape(arr, target, pad_value=0):
        
        channels_combined = []
    
        # print(arr.shape)
        # quit()
        for channel in arr:
          
          row_diff = target[0] - channel.shape[0]
        
        
          column_diff = target[1] - channel.shape[1]
          pad_row = (row_diff // 2, row_diff // 2)
          if row_diff % 2 != 0:
              pad_row = (row_diff // 2 + 1, row_diff // 2)
        
         
          pad_column = (column_diff // 2, column_diff // 2) 
          if column_diff % 2 != 0:
              pad_column = (column_diff // 2 + 1, column_diff // 2)
          
          channels_combined.append(np.pad(channel,  (pad_row, pad_column), constant_values=pad_value))      
        # print(np.sum(np.stack(channels_combined), axis=0).shape)
        # quit()  
        return np.sum(np.stack(channels_combined), axis=0)
    

    def channels_pad_batch(samples: ndarray, example: ndarray, axis: tuple, pad_value=0):
     
    #  samples = np.vstack(samples)
    #  example = np.vstack(example)
    #  print(samples.shape)
    #  print(example.shape)
    #  quit()
    # "Pad a list of arrays to the same shape (max of each dimension)."
     return np.array([Utility.channels_pad_to_shape(s, (example.shape[axis[0]], example.shape[axis[1]]), pad_value) for s in samples])
    




    class Agent:
        def __init__(self):
           super(self).__init__()

        
        def policy_loss(probs, action, reward):
            prob = probs[action]
            log_prob = np.log(prob)

            onehot = np.zeros(probs)
            onehot[action] = 1

            grad = reward * (probs - onehot)
            return -log_prob * reward, grad
