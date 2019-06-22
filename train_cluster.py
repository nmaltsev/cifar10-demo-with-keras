import os, sys, time
import numpy as np
import cPickle as pickle
from mpi4py import MPI

import keras
from keras.utils import np_utils

from model import createCifarCNN, compile, loadModel
from prepare_data import load_data, load_train, predict
from libs.configure import configureHardware

# https://stackoverflow.com/questions/21088420/mpi4py-send-recv-with-tag


if __name__ == '__main__':
	configureHardware(num_cores=4, num_CPU=1, num_GPU=0)
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()
	status = MPI.Status() 

	train_path =  'data/train_x_{}.pkl'.format(rank)
	test_path = 'data/test_x.pkl'

	if rank == 0:
		print('STEP 0: Load model')
		model = compile(createCifarCNN())	
		# nb_train_items, input_dim, nb_classes = find_input_dim(file_dir+train_name)
        # model = build_model(input_dim, nb_classes)
		## Since keras 1.0.7
		#model.save('master_model.h5')
		model_json = model.to_json()   # transform the model in json format
		model_weights = model.get_weights()    # returns a list of all weight tensors in the model as Numpy arrays.
	else:
		model_json=None
		model_weights=None

	if rank == 0:
		print('STEP 1: Distribute model')
		for i in range(1, size):
			# comm.send(model_json, dest=i)
			req = comm.isend(model_json, dest=i)
			req.wait()
	else:
		# received_model_json = comm.irecv(source=0)
		req = comm.irecv(source=0)
		received_model_json = req.wait()
		model = compile(loadModel(received_model_json))
		print('Worker: {} model is ready'.format(rank))
	comm.barrier()

	if rank == 0:
		print('STEP2 : Master brodcast weights to all slave')
		for i in range(1, size):
			req = comm.send(model_weights, dest=i)
			print('Send to {}'.format(i))
		
	else:
		received_model_json = comm.recv(source=0)
		model.set_weights(received_model_json)
		print('Worker: {} weights are ready'.format(rank))
	