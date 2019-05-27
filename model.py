import os

import keras.models as KM
import keras.layers as KL
import keras.layers.convolutional as KLC
import keras.layers.normalization as KLN
import keras.optimizers as KO
from keras.utils import np_utils


class Model(KM.Sequential):
	def __init__(self, model_name, nb_classes=10, nb_epoch=1,
							 batch_size=16, img_row=32, img_column=32, img_channel=3):
		super(Model, self).__init__()
		self.model_name = model_name
		self.batch_size = batch_size
		self.nb_classes = nb_classes
		self.nb_epoch = nb_epoch
		self.img_row = img_row
		self.img_column = img_column
		self.img_channel = img_channel
		self.x_train = []
		self.y_train = []
		self.x_test = []
		self.y_test = []
		self.prepared = False
		self.trained = False

		# Convolution layers
		self.add(KLC.ZeroPadding2D(input_shape=(3, 32, 32), padding=(2, 2)))
		self.add(KLC.Convolution2D(
				64, 5, 5, border_mode='valid', activation='relu'))
		self.add(KLN.BatchNormalization())
		self.add(KLC.MaxPooling2D(
				pool_size=(3, 3), strides=(2, 2), border_mode='same'))

		self.add(KLC.ZeroPadding2D(padding=(2, 2)))
		self.add(KLC.Convolution2D(64, 5, 5, border_mode='valid', activation='relu'))
		self.add(KLN.BatchNormalization())
		self.add(KLC.MaxPooling2D(
				pool_size=(3, 3), strides=(2, 2), border_mode='same'))

		self.add(KLC.ZeroPadding2D(padding=(2, 2)))
		self.add(KLC.Convolution2D(128, 5, 5, border_mode='valid', activation='relu'))
		self.add(KLN.BatchNormalization())
		self.add(KLC.MaxPooling2D(
				pool_size=(3, 3), strides=(2, 2), border_mode='same'))

		# Fully connected layer
		self.add(KL.Flatten())
		self.add(KL.Dense(1000))
		self.add(KL.Dropout(0.25))
		self.add(KL.Activation('relu'))
		self.add(KL.Dense(self.nb_classes))
		self.add(KL.Activation('softmax'))

		self.optimizer = KO.Adam(lr=0.0001, epsilon=1e-8)

	def set_train_data(self, x, y):
		#input x is 2d matrix. y is vector
		self.y_train = np_utils.to_categorical(y)
		self.x_train = x
		print('Successfully set train data. Train model.')
		self.prepared = True

	def save_m(self, file_path='./'):
		# super(Model, self).save(os.path.join(file_path, self.model_name))
		# serialize model to JSON
		if self.loss is None:
			raise Exception('Loss function is not defined. Please execute `model.compile(...)`')
		
		path_s = os.path.join(file_path, self.model_name)
		model_json = self.to_json()
		with open(path_s + '.json', 'w') as json_file:
			json_file.write(model_json)
		# serialize weights to HDF5
		self.save_weights(path_s + '.h5')
		print('Model saved to disk `%s`' %(path_s))

	def train(self, validation_data=None):
		if self.prepared:
			self.compile(
				loss='categorical_crossentropy',
				optimizer=self.optimizer,
				metrics=['accuracy']
			)
			self.fit(
				self.x_train,
				self.y_train,
				batch_size=self.batch_size,
				nb_epoch=self.nb_epoch,
				#  validation_split=0.1,
				validation_data=validation_data,
				shuffle=True
			) # ,show_accuracy=True
		else:
			print('Please set data to train model.')

# https://machinelearningmastery.com/save-load-keras-deep-learning-models/
def importModel(path_s):
	# load json and create model
	json_file = open(path_s + '.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	
	print('loaded_model_json')
	print(loaded_model_json)
	loaded_model = KM.model_from_json(loaded_model_json)
			# import json
			# loaded_model = KM.model_from_json(json.dumps(loaded_model_json))
	# load weights into new model
	loaded_model.load_weights(path_s + '.h5')
	print('Loaded model from disk')
	return loaded_model  
	# evaluate loaded model on test data
	# loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	
	# score = loaded_model.evaluate(X, Y, verbose=0)
	# print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

"""
Input config object:
[{u'class_name': u'ZeroPadding2D', u'config': {u'padding': [2, 2], u'batch_input_shape': [None, 3, 32, 32], u'trainable': True, u'name': u'zeropadding2d_1', u'input_dtype': u'float32'}}, {u'class_name': u'Convolution2D', u'config': {u'W_constraint': None, u'b_constraint': None, u'name': u'convolution2d_1', u'activity_regularizer': None, u'trainable': True, u'dim_ordering': u'th', u'nb_col': 5, u'subsample': [1, 1], u'init': u'glorot_uniform', u'bias': True, u'nb_filter': 64, u'b_regularizer': None, u'W_regularizer': None, u'nb_row': 5, u'activation': u'relu', u'border_mode': u'valid'}}, {u'class_name': u'BatchNormalization', u'config': {u'name': u'batchnormalization_1', u'epsilon': 1e-06, u'trainable': True, u'mode': 0, u'momentum': 0.9, u'axis': -1}}, {u'class_name': u'MaxPooling2D', u'config': {u'name': u'maxpooling2d_1', u'trainable': True, u'dim_ordering': u'th', u'pool_size': [3, 3], u'strides': [2, 2], u'border_mode': u'same'}}, {u'class_name': u'ZeroPadding2D', u'config': {u'padding': [2, 2], u'trainable': True, u'name': u'zeropadding2d_2'}}, {u'class_name': u'Convolution2D', u'config': {u'W_constraint': None, u'b_constraint': None, u'name': u'convolution2d_2', u'activity_regularizer': None, u'trainable': True, u'dim_ordering': u'th', u'nb_col': 5, u'subsample': [1, 1], u'init': u'glorot_uniform', u'bias': True, u'nb_filter': 64, u'b_regularizer': None, u'W_regularizer': None, u'nb_row': 5, u'activation': u'relu', u'border_mode': u'valid'}}, {u'class_name': u'BatchNormalization', u'config': {u'name': u'batchnormalization_2', u'epsilon': 1e-06, u'trainable': True, u'mode': 0, u'momentum': 0.9, u'axis': -1}}, {u'class_name': u'MaxPooling2D', u'config': {u'name': u'maxpooling2d_2', u'trainable': True, u'dim_ordering': u'th', u'pool_size': [3, 3], u'strides': [2, 2], u'border_mode': u'same'}}, {u'class_name': u'ZeroPadding2D', u'config': {u'padding': [2, 2], u'trainable': True, u'name': u'zeropadding2d_3'}}, {u'class_name': u'Convolution2D', u'config': {u'W_constraint': None, u'b_constraint': None, u'name': u'convolution2d_3', u'activity_regularizer': None, u'trainable': True, u'dim_ordering': u'th', u'nb_col': 5, u'subsample': [1, 1], u'init': u'glorot_uniform', u'bias': True, u'nb_filter': 128, u'b_regularizer': None, u'W_regularizer': None, u'nb_row': 5, u'activation': u'relu', u'border_mode': u'valid'}}, {u'class_name': u'BatchNormalization', u'config': {u'name': u'batchnormalization_3', u'epsilon': 1e-06, u'trainable': True, u'mode': 0, u'momentum': 0.9, u'axis': -1}}, {u'class_name': u'MaxPooling2D', u'config': {u'name': u'maxpooling2d_3', u'trainable': True, u'dim_ordering': u'th', u'pool_size': [3, 3], u'strides': [2, 2], u'border_mode': u'same'}}, {u'class_name': u'Flatten', u'config': {u'trainable': True, u'name': u'flatten_1'}}, {u'class_name': u'Dense', u'config': {u'W_constraint': None, u'b_constraint': None, u'name': u'dense_1', u'activity_regularizer': None, u'trainable': True, u'init': u'glorot_uniform', u'bias': True, u'input_dim': None, u'b_regularizer': None, u'W_regularizer': None, u'activation': u'linear', u'output_dim': 1000}}, {u'class_name': u'Dropout', u'config': {u'p': 0.25, u'trainable': True, u'name': u'dropout_1'}}, {u'class_name': u'Activation', u'config': {u'activation': u'relu', u'trainable': True, u'name': u'activation_1'}}, {u'class_name': u'Dense', u'config': {u'W_constraint': None, u'b_constraint': None, u'name': u'dense_2', u'activity_regularizer': None, u'trainable': True, u'init': u'glorot_uniform', u'bias': True, u'input_dim': None, u'b_regularizer': None, u'W_regularizer': None, u'activation': u'linear', u'output_dim': 10}}, {u'class_name': u'Activation', u'config': {u'activation': u'softmax', u'trainable': True, u'name': u'activation_2'}}]
relative code:
       print('CONFIG')
        print(config)
        for layer_data in config['layers']:
            process_layer(layer_data)
Reason - there no Layers property in config.
The bug while saving model!

TODO: find on github/google model json file and compare with dumped!

"""
