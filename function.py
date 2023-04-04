import matplotlib.pyplot as plt

def printLearningCurve(train_history, test_history):
    #Plot courbe d'apprentissage
    plt.figure(figsize=(16, 4))

    plt.subplot(1, 4, 1)
    plt.plot(train_history[:, 0], label='Train Loss')
    plt.legend()
    plt.subplot(1, 4, 2)
    plt.plot(train_history[:, 1], label='Train Accuracy')
    plt.legend()

    plt.subplot(1, 4, 3)
    plt.plot(test_history[:, 0], label='Test Loss')
    plt.legend()
    plt.subplot(1, 4, 4)
    plt.plot(test_history[:, 1], label='Test Accuracy')
    plt.legend()

    plt.show()