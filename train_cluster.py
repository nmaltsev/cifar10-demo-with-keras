import os, sys, time
import numpy as np
import cPickle as pickle
from mpi4py import MPI

import keras
from keras.utils import np_utils

from model import createCifarCNN, compile, loadModel, getDataGenerator
from prepare_data import load_data, load_train, predict
from libs.configure import configureHardware
from dataset import restoreDatasetChunk

# https://stackoverflow.com/questions/21088420/mpi4py-send-recv-with-tag

def average_weights(all_weights):
	new_weights = []
	for weights_list_tuple in zip(*all_weights):
		# new_weights.append([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)])
		new_weights.append(np.array([np.array(weights_).mean(axis=0) for weights_ in zip(*weights_list_tuple)]))
	return new_weights


if __name__ == '__main__':
	configureHardware(num_cores=4, num_CPU=1, num_GPU=0)
	comm = MPI.COMM_WORLD
	size = comm.Get_size()
	rank = comm.Get_rank()
	status = MPI.Status() 

	train_path =  'data/train_x_{}.pkl'.format(rank)
	test_path = 'data/test_x.pkl'
	train_x, train_y = restoreDatasetChunk(rank)
	datagen = getDataGenerator(train_x)

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

	for epoch in range(2):
		print('Epoch {}'.format(epoch))

		model.fit_generator(
			datagen.flow(train_x, train_y, batch_size=16),
			len(train_x),
			# nb_epoch=1,
			1,
			# TODO
			# validation_data=validation_data,
		)

		update_weights =  model.get_weights()
		# model.save_weights('weights_r{}_e{}.h5'.format(rank,epoch))
		"""
		if rank == 0:
			w = [
				# model_weights, # maybe cause troubles
				update_weights
			]
			w1 = comm.recv(source=MPI.ANY_SOURCE, status=status)
			source = status.Get_source()
			w.append(w1)
			model_weights=average_weights(w)
			comm.send(model_weights, dest=source)
		else:
			comm.send(update_weights, dest=0)
			received_model_weights = comm.recv(source=0)
			print('received_model_weights')
			model.set_weights(received_model_weights)
		print(rank, "after send/recept from slave to master")
		"""
		all_received_weights = comm.gather(update_weights, root=0)
		print(rank, "after gather weights")

		if rank == 0:
			print('Master received all weights {}'.format(len(all_received_weights)))
			model_weights = average_weights(all_received_weights)

		if rank==0:
			print(rank, "Model Evaluation")
			model.set_weights(model_weights)
			## TODO: undump
			# dumpMatrix(
			# 	(x_norm[1], raw_test_y),
			# 	'data/test_chunk.pkl'
			# )

			#score =  received_model.evaluate_generator(read_file_chunk(file_dir+test_name, chunksize, nb_classes),
                                     steps=nb_test_items//chunksize)
	
"""
Traceback (most recent call last):
  File "train_cluster.py", line 104, in <module>
    model.set_weights(received_model_weights)
  File "/usr/local/lib/python2.7/dist-packages/keras/models.py", line 286, in set_weights
    layer.set_weights(weights[:nb_param])
  File "/usr/local/lib/python2.7/dist-packages/keras/engine/topology.py", line 852, in set_weights
    if pv.shape != w.shape:
AttributeError: 'list' object has no attribute 'shape'
"""
