import matplotlib.pyplot as plt
import h5py
import numpy as np
import cv2

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
    X_train = np.append(X_train, cv2.rotate(np.array(train_dataset["X_train"][:]), cv2.ROTATE_180), axis = 0)
    y_train = np.array(train_dataset["Y_train"][:]) # Train labels
    y_train = np.append(y_train, train_dataset["Y_train"][:], axis = 0)

    test_dataset = h5py.File('datasets/testset.hdf5', "r")
    X_test = np.array(test_dataset["X_test"][:]) # Test features
    X_test = np.append(X_test, cv2.rotate(np.array(test_dataset["X_test"][:]), cv2.ROTATE_180), axis = 0)
    y_test = np.array(test_dataset["Y_test"][:]) # Test labels
    y_test = np.append(y_test, cv2.rotate(test_dataset["Y_test"][:], cv2.ROTATE_180), axis = 0)

    # plt.figure(figsize=(16, 8))
    # j = 1
    # for i in range(1, 400, 20):
    #     plt.subplot(4, 5, j)
    #     plt.imshow(X_test[i], cmap='gray')
    #     plt.title(y_test[i])
    #     plt.tight_layout()
    #     j += 1
    # plt.show()

    return X_train, y_train, X_test, y_test
