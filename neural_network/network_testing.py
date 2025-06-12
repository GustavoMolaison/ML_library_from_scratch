from layers.hugo_dense import Dense_Layer
from models.Sequential import Hugo
from layers.hugo_conv import Conv_layer
import numpy as np
from utils.hugo_utility import Utility as U
import matplotlib.pyplot as plt

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
X_train = X  
Y =  Y.reshape(-1,1)
y_train = Y
# # print(X_training.shape)
# print(Y_training.shape)
# quit()
import glob
import os
from PIL import Image
import kagglehub
from sklearn.model_selection import train_test_split

# kagglehub.login()

path = kagglehub.dataset_download("jcprogjava/handwritten-digits-dataset-not-in-mnist")
print("Path to dataset files:", path)



if os.path.exists('dataset.npz'):
    data = np.load('dataset.npz')
    X = data['X']
    y = data['y']

else:
        IMAGE_SIZE = (28, 28)  # Resize to match your CNN input
        numbers_paths = []
        labels = []
        y = []
        for dirpath, dirnames, filenames in os.walk(path):
           print('Current Path:', dirpath)
           print('Directories', dirnames)
    
           if  len(dirnames) == 1 and len(dirnames[0]) == 1:
              numbers_paths.append(os.path.join(dirpath, dirnames[0]))
              labels.append(int(dirnames[0]))


        images_paths = []
        for idx, num_paths in enumerate(numbers_paths):
          for num_path in os.listdir(num_paths):
            images_paths.append(os.path.join(num_paths, num_path))
            y.append(labels[idx])
        
        X = []
        for image_path in images_paths:
          
          img = Image.open(image_path)
          # print(img)
          # img = img.resize(IMAGE_SIZE) # Grayscale
          # img_array = np.array(img, dtype=np.float32) 
          img_array = np.array(img, dtype=np.float32) / 255.0
          # plt.imshow(img_array, cmap='gray')
          # plt.show()
          print(img_array)
          # quit()
          X.append(img_array)

        X = X = np.array(X).reshape(-1, 1, 28, 28)
        y = np.array(y)
        np.savez_compressed('dataset.npz', X=X, y=y)


# Split into train and test
X = X[ : 107730]
print(X.shape)
print(y.shape)
# quit()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train[: 100]
y_train_b = y_train[: 100]
y_train = U.one_hot_encoding(y_train_b, 10)
X_test = X_test[: 100]
y_test_b = y_test[: 100]
y_test = U.one_hot_encoding(y_test_b, 10)
# print(X_train[0])
# print(y_test)
# quit()
print(X_train.shape)
print(y_train.shape)
# quit()

#  Here is my model syntax we can add multiple layers to one model or run layers forward and backwards individually 
hugo = Hugo(loss = 'cross_entropy', update_method = 'SGD', clip_method = 'norm clipping', weight_initialization= 'he', dropout = False, lr = 0.01)

layer_conv = Conv_layer(model = hugo.model)
layer_conv.set_layer(param = (3,3), weight_initialization = 'he', activation_function= 'none')
hugo.model.add_layer(layer = layer_conv, dense = 1)

layer_I = Dense_Layer(model = hugo.model)
layer_I.set_layer(neurons_num=64, activation_function = 'tanh', weight_initialization= 'he')
hugo.model.add_layer(layer = layer_I, dense = 1)

layer_D = Dense_Layer(model = hugo.model)
layer_D.set_layer(neurons_num=64, activation_function = 'tanh', weight_initialization= 'he')
hugo.model.add_layer(layer = layer_D, dense = 1)

layer_O = Dense_Layer(model = hugo.model)
layer_O.set_layer(neurons_num = 10, activation_function = 'none', weight_initialization= 'he')
hugo.model.add_layer(layer = layer_O, dense = 1)

# hugo.set_layers(X = X_training, Y = Y_training,  model_nn = hugo.model,
#                  neurons_num = 64, density = 1,
#                  activation_functions = ['leaky relu','leaky relu','leaky relu'], lr_update_method = ['none','none','none'], 
#                  weight_initialization= [None, None, None])




loss_over_epochs_t, loss_over_epochs_v, output_t, output_v = hugo.run(model_nn = hugo.model, epochs = 100, X = X_train, Y = y_train, X_val = X_test, Y_val = y_test)











layers = hugo.model.layers
# for i  in range(len(layers)):
#     for w in range(len(layers[i].weights_ac_epo) - 1):
#      if (layers[i].weights_ac_epo[w] == layers[i].weights_ac_epo[w + 1]).all():
#         print('WEIGHTS ARE THE SAME')
   


# output_t = np.round(output_t).astype(int)
output_t = np.argmax(output_t, axis = 1)
accuracy_t = np.mean(output_t == y_train_b)

print(f'rounded output training: {output_t}')
print(f'training loss: {loss_over_epochs_t[-1]}')
print(f'training accuracy: {accuracy_t}')
print('\n')
# output_v = np.round(output_v).astype(int)
output_v = np.argmax(output_v, axis = 1)
accuracy_v = np.mean(output_v == y_test_b)
print(f'rounded output validation: {output_v}')
print(f'VALIDATION loss: {loss_over_epochs_v[-1]}')
print(f'test accuracy: {accuracy_v}')
