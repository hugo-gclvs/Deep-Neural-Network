import matplotlib.pyplot as plt

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