import os, sys, time
import numpy as np
import cPickle as pickle
from mpi4py import MPI

import keras
from keras.utils import np_utils

from model import createCifarCNN, compile, loadModel, getDataGenerator
from prepare_data import load_data, load_train, predict
from libs.configure import configureHardware
from libs.timer import Timer
from dataset import restoreDatasetChunk, restoreTestDataset


def test():
    model = compile(createCifarCNN())
    # model.set_weights(model_weights)
    model.load_weights('weights_r{}_e{}.h5'.format(0, 2))
    
    test_x, test_y = restoreTestDataset()
    train_x, train_y = restoreDatasetChunk(0)
    batch_size = 32
    traingen = getDataGenerator(test_x)

    print('Test_x len', len(test_x))
    print(test_x.shape, test_y.shape, train_x.shape, train_y.shape, test_y[0], train_y[0])
    # print(traingen.flow(test_x, test_y, batch_size=batch_size))
    
    # score = model.evaluate_generator(
    #     traingen.flow(test_x, test_y, batch_size=batch_size),
    #     len(test_x)//batch_size
    # )
    score = model.evaluate(test_x, test_y)
    
    # score = model.evaluate_generator(
    #     traingen.flow(train_x, train_y, batch_size=batch_size),
    #     len(train_x)//batch_size
    # )
    # score = model.evaluate(train_x, train_y)

    print('Score:')
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    test()

