import numpy as np


from model import createCifarCNN, Trainer
from prepare_data import load_data, load_train, predict
from libs.configure import configureHardware
from libs.timer import Timer
from libs.training_plot import TrainingPlot
from dataset import restoreDataset 

if __name__ == '__main__':
	USE_BATCH_TRAIN = True
	configureHardware(num_cores=4, num_CPU=1, num_GPU=0)
	timer = Timer().start()

	train_x, train_y, valid_x, valid_y, test_x, test_y = restoreDataset()
	plot_losses = TrainingPlot()

	model = Trainer(
		'cifar1.model', 
		createCifarCNN(),
		batch_size=64, 
		# nb_epoch=2
		nb_epoch=20
	)
	print('[START train]')
	print(train_x.shape)
	model.set_train_data(train_x, train_y)
	
	if USE_BATCH_TRAIN:
		model.train_gen(validation_data=(valid_x, valid_y), callbacks=[plot_losses])
	else:
		model.train(validation_data=(valid_x, valid_y))
	
	timer.stop().note('Training time').start()

	model.save_m()
	predict(model.model, test_x, test_y)
	timer.stop().note('Prediction time')
	print('\n'.join(['{}: {}'.format(note, sec) for (note, sec) in timer.stack]))
	
