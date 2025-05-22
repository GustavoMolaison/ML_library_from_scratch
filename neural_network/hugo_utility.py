import numpy as np

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
    
    def cross_entropy_loss(x, y, eps=1e-15):
       
       # Clip to prevent log(0)
       x = np.clip(x, eps, 1 - eps)
      
       # Apply element-wise loss
       loss = -np.sum(y * np.log(x), axis=1)
    
       return np.mean(loss)

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
      

