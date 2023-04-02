import matplotlib.pyplot as plt
import h5py
import numpy as np


def printLearningCruve(train_history):
    #Plot courbe d'apprentissage
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_history[:, 0], label='Train Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(train_history[:, 1], label='Train Accuracy')
    plt.legend()
    plt.show()


def load_data():
    train_dataset = h5py.File('datasets/trainset.hdf5', "r")
    X_train = np.array(train_dataset["X_train"][:]) # Train features
    y_train = np.array(train_dataset["Y_train"][:]) # Train labels

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # Test features
    y_test = np.array(test_dataset["Y_test"][:]) # Test labels
    
    return X_train, y_train, X_test, y_test