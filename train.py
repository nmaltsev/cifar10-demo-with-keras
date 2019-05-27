import numpy as np
import cPickle as pickle

from model import Model
from prepare_data import load_data, load_train, predict
from keras.utils import np_utils

from configure import configureHardware

if __name__ == '__main__':
	configureHardware(num_cores=4, num_CPU=1, num_GPU=0)
	
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


	model = Model('cifar1.model')
	print('[START train]')
	model.set_train_data(train_x, train_y)
	model.train(validation_data=(valid_x, valid_y))
	model.save_m()
	predict(model, test_x, test_y)
	
	
