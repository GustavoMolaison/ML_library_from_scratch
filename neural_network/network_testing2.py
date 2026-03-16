import numpy as np

from hugo_flow.layers.hugo_dense import Dense_Layer
from hugo_flow.models.Sequential import Hugo

def get_continuous_xor(n_points=1000):
    X = np.random.uniform(-1, 1, (n_points, 2))
    # Jeśli znaki x1 i x2 są różne -> klasa 1, jeśli takie same -> klasa 0
    y = (np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)).astype(int).reshape(-1, 1)
    return X, y

X_train, y_train = get_continuous_xor(2000)
X_test, y_test = get_continuous_xor(2000)

hugo = Hugo(loss = 'mse', weight_initialization= 'he', dropout = False, lr = 0.001)
layer_I = Dense_Layer(model = hugo.model)
layer_I.set_layer(neurons_num= 16, activation_function = 'sigmoid', weight_initialization= 'he')
hugo.model.add_layer(layer = layer_I, dense = 1)


layer_2 = Dense_Layer(model = hugo.model)
layer_2.set_layer(neurons_num= 1, activation_function = 'sigmoid', weight_initialization= 'he')
hugo.model.add_layer(layer = layer_2, dense = 1)

loss_over_epochs_t, loss_over_epochs_v, output_t, output_v = hugo.run(model_nn = hugo.model, epochs = 100, X = X_train, Y = y_train, X_val = X_test, Y_val = y_test)