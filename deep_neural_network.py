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

def deep_neural_network(X_train, y_train, X_test, y_test, hidden_layers = 16, learning_rate = 0.01, epochs = 3000):
    
    #Train set
    #Init parameters
    dimensions = list()
    dimensions.insert(0, hidden_layers)
    dimensions.insert(0, X_train.shape[0])
    dimensions.append(y_train.shape[0])
    print(dimensions)
    np.random.seed(1)
    parametres = initialisation(dimensions)

    #Tableau numpy contenant les futures accuracy et log_loss
    training_history = np.zeros((int(epochs), 2))
    test_history = np.zeros((int(epochs), 2))

    C = len(parametres) // 2

    #Gradient descent
    for i in tqdm(range(epochs)):

        activations = forward_propagation(X_train, parametres)
        gradients = back_propagation(y_train, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)
        Af = activations['A' + str(C)]

        # calcul du log_loss et de l'accuracy
        training_history[i, 0] = (log_loss(y_train.flatten(), Af.flatten()))
        y_pred = predict(X_train, parametres)
        training_history[i, 1] = (accuracy_score(y_train.flatten(), y_pred.flatten()))

        # Validation set
        activations = forward_propagation(X_test, parametres)
        Af = activations['A' + str(C)]

        # calcul du log_loss et de l'accuracy
        
        test_history[i, 0] = (log_loss(y_test.flatten(), Af.flatten()))
        y_pred = predict(X_test, parametres)
        test_history[i, 1] = (accuracy_score(y_test.flatten(), y_pred.flatten()))

    return training_history, test_history


def __main__():
    #Create train Dataset
    X_train, y_train = make_circles(n_samples=1000, noise=0.1, factor=0.3, random_state=0)
    X_train = X_train.T
    y_train = y_train.reshape((1, y_train.shape[0]))
    # show dataset
    # plt.scatter(X_train[0, :], X_train[1, :], c=y_train, cmap='summer')
    # plt.show()


    #Create test Dataset
    X_test, y_test = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=1)
    X_test = X_test.T
    y_test = y_test.reshape((1, y_test.shape[0]))

    # train_history, test_history = deep_neural_network(X_train, y_train, X_test, y_test, hidden_layers = 64, learning_rate = 0.1, epochs = 500)

    # print('Accuracy on train set: ', round(train_history[-1, 1], 5))
    # print("Lost on train set: ", round(train_history[-1, 0], 5))
    # print('Accuracy on test set: ', round(test_history[-1, 1], 5))
    # print("Lost on test set: ", round(test_history[-1, 0], 5))

    # plt.figure(figsize=(12, 4))
    # #Print train Dataset
    # plt.subplot(1, 2, 1)
    # plt.scatter(X_train[0, :], X_train[1, :], c=y_train, cmap='summer')
    # plt.title('TRAIN DATASET')

    # #Print test Dataset
    # plt.subplot(1, 2, 2)
    # plt.scatter(X_test[0, :], X_test[1, :], c=y_test, cmap='cool')
    # plt.title('TEST DATASET')

    # plt.show()

    # printLearningCurve(train_history, test_history)

__main__()