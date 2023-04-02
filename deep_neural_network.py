import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles
from sklearn.metrics import accuracy_score, log_loss
from tqdm import tqdm
from init import initialisation
from forward_propa import forward_propagation
from back_propa import back_propagation
from update import update
from predict import predict
from function import *

def deep_neural_network(X_train, y_train, X_test, y_test, hidden_layers = (16, 16, 16), learning_rate = 0.001, n_iter = 3000):
    
    #Train set
    #Init parameters
    dimensions = list(hidden_layers)
    dimensions.insert(0, X_train.shape[0])
    dimensions.append(y_train.shape[0])
    np.random.seed(1)
    parametres = initialisation(dimensions)

    #Tableau numpy contenant les futures accuracy et log_loss
    training_history = np.zeros((int(n_iter), 2))

    C = len(parametres) // 2

    #Gradient descent
    for i in tqdm(range(n_iter)):

        activations = forward_propagation(X_train, parametres)
        gradients = back_propagation(y_train, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)
        Af = activations['A' + str(C)]

        # calcul du log_loss et de l'accuracy
        training_history[i, 0] = (log_loss(y_train.flatten(), Af.flatten()))
        y_pred = predict(X_train, parametres)
        training_history[i, 1] = (accuracy_score(y_train.flatten(), y_pred.flatten()))

    #Test set
    # gradient descent
    for i in tqdm(range(n_iter)):

        activations = forward_propagation(X_test, parametres)
        Af = activations['A' + str(C)]

        # calcul du log_loss et de l'accuracy
        y_test = predict(X_test, parametres)

    return training_history, y_test


def __main__():
    
    X_train, y_train, X_test, y_test = load_data()

    X_train_reshaped = (X_train.reshape(X_train.shape[0], -1) / X_train.max())
    X_test_reshaped = (X_test.reshape(X_test.shape[0], -1) / X_train.max())
    X_train_reshaped = X_train_reshaped.T
    X_test_reshaped = X_test_reshaped.T

    y_train = y_train.reshape((1, y_train.shape[0]))
    y_test = y_test.reshape((1, y_test.shape[0]))

    train_history, y_pred = deep_neural_network(X_train_reshaped, y_train, X_test_reshaped, y_test, hidden_layers = (16, 16, 16), learning_rate = 0.1, n_iter = 3000)

    print('Accuracy on train set: ', round(train_history[-1, 1], 5))
    print("Lost on train set: ", round(train_history[-1, 0], 5))

    print(y_pred)
    printLearningCruve(train_history)

__main__()