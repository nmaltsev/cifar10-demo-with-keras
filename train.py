import numpy as np


from model import createCifarCNN, Trainer
from prepare_data import load_data, load_train, predict
from libs.configure import configureHardware
from libs.timer import Timer
from libs.training_plot import TrainingPlot
from dataset import restoreDataset 
from libs.early_stopping_by_loss import EarlyStoppingByLoss

if __name__ == '__main__':
	USE_BATCH_TRAIN = True
	configureHardware(num_cores=4, num_CPU=1, num_GPU=0)
	timer = Timer().start()

	train_x, train_y, valid_x, valid_y, test_x, test_y = restoreDataset()

	trainer = Trainer(
		'cifar1.model', 
		createCifarCNN(),
		batch_size=64, 
		# nb_epoch=1
		nb_epoch=20
	)
	trainer.model.summary()
	trainer.set_train_data(train_x, train_y)
	
	if USE_BATCH_TRAIN:
		trainer.train_gen(validation_data=(valid_x, valid_y), callbacks=[
			TrainingPlot('plot.data'),
			EarlyStoppingByLoss(0.5)
		])
	else:
		trainer.train(validation_data=(valid_x, valid_y))
	
	timer.stop().note('Training time').start()

	trainer.save_m()
	predict(trainer.model, test_x, test_y)
	timer.stop().note('Prediction time')
	print('\n'.join(['{}: {}'.format(note, sec) for (note, sec) in timer.stack]))
	"""
Epoch 13/20
44992/45000 [============================>.] - ETA: 0s - loss: 0.5030 - acc: 0.8251('\nEE ', 0.5030798812654284, ' ', 0.5)
{'acc': 0.8250444444444445, 'loss': 0.5030798812654284, 'val_acc': 0.8022, 'val_loss': 0.5652871282100678}
45000/45000 [==============================] - 1579s - loss: 0.5031 - acc: 0.8250 - val_loss: 0.5653 - val_acc: 0.8022
Epoch 14/20
44992/45000 [============================>.] - ETA: 0s - loss: 0.4724 - acc: 0.8331('\nEE ', 0.4723822776052687, ' ', 0.5)
{'acc': 0.8331555555555555, 'loss': 0.4723822776052687, 'val_acc': 0.809, 'val_loss': 0.5323268394470215}
45000/45000 [==============================] - 1518s - loss: 0.4724 - acc: 0.8332 - val_loss: 0.5323 - val_acc: 0.8090
Model saved to disk `./cifar1.model`
10000/10000 [==============================] - 148s     
accuracy:80.89
Training time: 21809
Prediction time: 149

	"""
