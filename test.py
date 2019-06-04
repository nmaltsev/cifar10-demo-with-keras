import numpy as np
import cPickle as pickle

from model import importModel, createCifarCNN, Trainer
# load_data, load_train, 
from prepare_data import predict
from keras.utils import np_utils

if __name__ == '__main__':
	with open('data/image_norm_zca.pkl', 'rb') as f:
		images = pickle.load(f)
		index = np.random.permutation(len(images['train']))
		train_index = index[:-5000]
		valid_index = index[-5000:]
		train_x = images['train'][train_index].reshape((-1, 3, 32, 32))
		valid_x = images['train'][valid_index].reshape((-1, 3, 32, 32))
		test_x = images['test'].reshape((-1, 3, 32, 32))

	with open('data/label.pkl', 'rb') as f:
		labels = pickle.load(f)
		train_y = labels['train'][train_index]
		valid_y = labels['train'][valid_index]
		valid_y = np_utils.to_categorical(valid_y)
		test_y = labels['test']

	# It has troubles with parsing JSON
	# g: Keras model_from_json list indices must be integers, not str
	model = importModel('cifar1.model')
	
	# ~ model = Trainer('cifar2.model', createCifarCNN())
	# ~ model.load_weights('cifar1.model.h5')
	
	print('[START predict]')
	predict(model, test_x, test_y)
