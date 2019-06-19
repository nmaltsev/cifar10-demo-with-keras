import os, sys, time
import numpy as np
import cPickle as pickle
from mpi4py import MPI

import keras
from keras.utils import np_utils

from model import createCifarCNN, Trainer
from prepare_data import load_data, load_train, predict
from configure import configureHardware

if __name__ == '__main__':
	configureHardware(num_cores=4, num_CPU=1, num_GPU=0)
