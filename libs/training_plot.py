import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
import keras

class TrainingPlot(keras.callbacks.Callback):
    def __init__(self, log_path):
        super(keras.callbacks.Callback, self).__init__()
        self.log_path = log_path

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    def on_train_end(self, logs={}):
        # Clear the previous plot
        # clear_output(wait=True)
        N = np.arange(0, len(self.losses))
        
        # You can chose the style of your preference
        # print(plt.style.available) to see the available options
        plt.style.use("seaborn")
        
        # Plot train loss, train acc, val loss and val acc against epochs passed
        plt.figure()
        plt.plot(N, self.losses, label = "train_loss")
        plt.plot(N, self.acc, label = "train_acc")
        plt.plot(N, self.val_losses, label = "val_loss")
        plt.plot(N, self.val_acc, label = "val_acc")
        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend()
        plt.show()
        # libpng-dev must be installed
        # plt.savefig('training_plot')
    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):
        
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        
        with open(self.log_path, 'wb') as f:
            pickle.dump({
                'epoch': epoch,
                'losses': self.losses,
                'acc': self.acc,
                'val_losses': self.val_losses,
                'val_acc': self.val_acc
            }, f, pickle.HIGHEST_PROTOCOL)
