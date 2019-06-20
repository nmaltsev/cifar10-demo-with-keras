import os, sys, time
import numpy as np
import cPickle as pickle
from mpi4py import MPI

import keras
from keras.utils import np_utils

from model import createCifarCNN, Trainer
from prepare_data import load_data, load_train, predict
from libs.configure import configureHardware

if __name__ == '__main__':
	configureHardware(num_cores=4, num_CPU=1, num_GPU=0)
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()
	status = MPI.Status() 
	
	
	if rank == 0:
		from dataset import prepare_dataset_chunks
		print('Rank', rank)
		## Path to dataset
		dataset_path = '/root/tfplayground/datasets/cifar-10-batches-py'
		## destination path for prepared dataset chunks
		destination_path = 'dataset_chunks'
		prepare_dataset_chunks(size - 1, dataset_path, destination_path)
	else:
		print('Rank', rank)
