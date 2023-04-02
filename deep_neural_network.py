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

def deep_neural_network(X, y, hidden_layers = (16, 16, 16), learning_rate = 0.001, n_iter = 3000):
    
    # initialisation parametres
    dimensions = list(hidden_layers)
    dimensions.insert(0, X.shape[0])
    dimensions.append(y.shape[0])
    np.random.seed(1)
    parametres = initialisation(dimensions)

    # tableau numpy contenant les futures accuracy et log_loss
    training_history = np.zeros((int(n_iter), 2))

    C = len(parametres) // 2

    # gradient descent
    for i in tqdm(range(n_iter)):

        activations = forward_propagation(X, parametres)
        gradients = back_propagation(y, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)
        Af = activations['A' + str(C)]

        # calcul du log_loss et de l'accuracy
        training_history[i, 0] = (log_loss(y.flatten(), Af.flatten()))
        y_pred = predict(X, parametres)
        training_history[i, 1] = (accuracy_score(y.flatten(), y_pred.flatten()))

    # Plot courbe d'apprentissage
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(training_history[:, 0], label='train loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(training_history[:, 1], label='train acc')
    plt.legend()
    plt.show()

    return training_history


def __main__():
    #Create train Dataset
    X_train, y_train = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=0)
    X_train = X_train.T
    y_train = y_train.reshape((1, y_train.shape[0]))

    #Create test Dataset
    X_test, y_test = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=1)
    X_test = X_test.T
    y_test = y_test.reshape((1, y_test.shape[0]))

    #Print train Dataset
    plt.scatter(X_train[0, :], X_train[1, :], c=y_train, cmap='summer')
    plt.title('TRAIN DATASET')
    plt.show()

    #Print test Dataset
    plt.scatter(X_test[0, :], X_test[1, :], c=y_test, cmap='summer')
    plt.title('TEST DATASET')
    plt.show()

    deep_neural_network(X_train, y_train, hidden_layers = (16, 16, 16), learning_rate = 0.1, n_iter = 3000)


__main__()