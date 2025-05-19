from nn_1 import Hugo, Dense_Layer
from hugo_conv import Conv_layer
import numpy as np

X = np.array(
    [  # Sample 0 - like "1"
    [    [[0, 1, 0],
         [0, 1, 0],
         [0, 1, 0],
         [0, 0, 0],
         [0, 1, 0],
         [0, 0, 0]]
    ],
    [  # Sample 1 - like "0"
        [[1, 1, 1],
         [1, 0, 1],
         [1, 1, 1],
         [0, 1, 0],
         [1, 0, 1],
         [0, 1, 0]]
    ],
    [  # Sample 2 - like "7"
        [[1, 1, 1],
         [0, 0, 1],
         [0, 1, 0],
         [0, 0, 1],
         [0, 1, 0],
         [1, 0, 0]]
    ]
], dtype=np.float32)

# Corresponding labels (Y): integers for classification
Y = np.array([1, 0, 7])
X_training = X  
Y =  Y.reshape(-1,1)
Y_training = Y
print(X_training.shape)
print(Y_training.shape)
# quit()





hugo = Hugo(loss = 'mse', weight_initialization= 'he', dropout = False)

layer_conv = Conv_layer(model = hugo.model)
layer_conv.set_layer(param = np.ones((3, 3)), weight_initialization = 'he', activation_function= 'tanh')
hugo.model.add_layer(layer = layer_conv, dense = 1)

layer_I = Dense_Layer(model = hugo.model)
layer_I.set_layer(neurons_num=64, activation_function = 'leaky relu', weight_initialization= 'he', lr_update_method= 'none')
hugo.model.add_layer(layer = layer_I, dense = 1)

layer_D = Dense_Layer(model = hugo.model)
layer_D.set_layer(neurons_num=64, activation_function = 'leaky relu', weight_initialization= 'he', lr_update_method= 'none')
hugo.model.add_layer(layer = layer_D, dense = 1)

layer_O = Dense_Layer(model = hugo.model)
layer_O.set_layer(neurons_num = Y_training.shape[1], activation_function = 'leaky relu', weight_initialization= 'he', lr_update_method= 'none')
hugo.model.add_layer(layer = layer_O, dense = 1)

# hugo.set_layers(X = X_training, Y = Y_training,  model_nn = hugo.model,
#                  neurons_num = 64, density = 1,
#                  activation_functions = ['leaky relu','leaky relu','leaky relu'], lr_update_method = ['none','none','none'], 
#                  weight_initialization= [None, None, None])

loss_over_epochs_t, loss_over_epochs_v, output = hugo.run(model_nn = hugo.model, epochs = 100, X = X_training, Y = Y_training,  )














layers = hugo.model.layers
# for i  in range(len(layers)):
#     for w in range(len(layers[i].weights_ac_epo) - 1):
#      if (layers[i].weights_ac_epo[w] == layers[i].weights_ac_epo[w + 1]).all():
#         print('WEIGHTS ARE THE SAME')
   

print(output)
output = np.round(output).astype(int)
accuracy = np.mean(output == Y_training)
print(f'rounded output: {output}')
print(f'training loss: {loss_over_epochs_t[-1]}')
print(f'training accuracy: {accuracy}')
print(f'VALIDATION loss: {loss_over_epochs_v[-1]}\n')