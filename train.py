import numpy as np


from model import createCifarCNN, Trainer
from prepare_data import load_data, load_train, predict
from libs.configure import configureHardware
from dataset import restoreDataset 

if __name__ == '__main__':
	configureHardware(num_cores=4, num_CPU=1, num_GPU=1)

	train_x, train_y, valid_x, valid_y, test_x, test_y = restoreDataset()

	model = Trainer(
		'cifar1.model', 
		createCifarCNN(),
		batch_size=32, 
		nb_epoch=5
	)
	print('[START train]')
	print(train_x.shape)
	model.set_train_data(train_x, train_y)
	# model.train(validation_data=(valid_x, valid_y))
	model.train_gen(validation_data=(valid_x, valid_y))
	model.save_m()
	predict(model.model, test_x, test_y)
	
	