import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

# import keras

import cPickle as pickle

def load(path_s):
    with open(path_s, 'rb') as f:
        data = pickle.load(f)
    
    return data

def plot_accuracy(data):
    plt.style.use('seaborn')
    # Plot training & validation accuracy values
    plt.plot(data['acc'])
    plt.plot(data['val_acc'])

    plt.xticks(range(0, len(data['acc'])))

    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    plt.savefig('training_accuracy')

def plot_loss(data):
    plt.style.use('seaborn')
    plt.xticks(range(0, len(data['losses'])))
    # Plot training & validation loss values
    plt.plot(data['losses'])
    plt.plot(data['val_losses'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()    
    plt.savefig('training_losses')

def plot_grid(data):
    #plt.figure(1)
    plt.figure(1, figsize=(10, 6), dpi=80, facecolor='w', edgecolor='k')
    plt.style.use('seaborn')
    
    plt.subplot(211)
    plt.plot(data['acc'])
    plt.plot(data['val_acc'])
    plt.xticks(range(0, len(data['acc'])))
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    
    plt.subplot(212)
    plt.xticks(range(0, len(data['losses'])))
    # Plot training & validation loss values
    plt.plot(data['losses'])
    plt.plot(data['val_losses'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    #plt.subplots_adjust(hspace=0.5)
    plt.subplots_adjust(wspace=0, hspace=0.4)
    plt.show()    
    plt.savefig('training_grid')
        

def plot_general(data):
    print(data)
    N = np.arange(0, len(data['losses']))            
    # You can chose the style of your preference
    # print(plt.style.available) to see the available options
    plt.style.use("seaborn")

    # Plot train loss, train acc, val loss and val acc against epochs passed
    plt.figure()
    plt.plot(N, data['losses'], label = "train_loss")
    plt.plot(N, data['acc'], label = "train_acc")
    plt.plot(N, data['val_losses'], label = "val_loss")
    plt.plot(N, data['val_acc'], label = "val_acc")
    plt.title("Training Loss and Accuracy [Epoch {}]".format(data['epoch']))
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()
    # libpng-dev must be installed
    plt.savefig('training_general')

    # with open('plot.data', 'wb') as f:
    #     pickle.dump({
    #         'epoch': epoch,
    #         'losses': self.losses,
    #         'acc': self.acc,
    #         'val_losses': self.val_losses,
    #         'val_acc': self.val_acc
    #     }, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    data = load('plot.data')
    #plot_general(data)
    #plot_accuracy(data)
    #plot_loss(data)
    plot_grid(data)
    