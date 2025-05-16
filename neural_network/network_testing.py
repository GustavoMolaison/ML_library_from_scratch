from nn_1 import Hugo, Layer
from hugo_conv import convolutional_model, convolutional_input_output
import numpy as np

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





hugo = Hugo(loss = 'mse', weight_initialization= 'he', dropout = False)
layer_I = Layer(model = hugo.model)
layer_I.set_layer(input_features = X_training.shape[1], neurons_num=64, activation_function = 'leaky relu', weight_initialization= 'he', lr_update_method= 'none')
hugo.model.add_layer(layer = layer_I, dense = 1)
layer_D = Layer(model = hugo.model)
layer_D.set_layer(input_features = 64, neurons_num=64, activation_function = 'leaky relu', weight_initialization= 'he', lr_update_method= 'none')
hugo.model.add_layer(layer = layer_D, dense = 1)

# hugo.set_layers(X = X_training, Y = Y_training,  model_nn = hugo.model,
#                  neurons_num = 64, density = 1,
#                  activation_functions = ['leaky relu','leaky relu','leaky relu'], lr_update_method = ['none','none','none'], 
#                  weight_initialization= [None, None, None])

loss_over_epochs_t, loss_over_epochs_v, output = hugo.run(model_nn = hugo.model, epochs = 100, X = X_training, Y = Y_training,  )














layers = hugo.model.layers
for i  in range(len(layers)):
    for w in range(len(layers[i].weights_ac_epo) - 1):
     if (layers[i].weights_ac_epo[w] == layers[i].weights_ac_epo[w + 1]).all():
        print('WEIGHTS ARE THE SAME')
   

print(output)
output = np.round(output).astype(int)
accuracy = np.mean(output == Y_training)
print(f'training loss: {loss_over_epochs_t[-1]}')
print(f'training accuracy: {accuracy}')
print(f'VALIDATION loss: {loss_over_epochs_v[-1]}\n')