import tensorflow as tf
from keras import backend as K


def configureHardware(num_cores=4, num_CPU=1, num_GPU=1):
	gpu_options = tf.GPUOptions(
		per_process_gpu_memory_fraction=0.9,
		allow_growth=True
	)

	config = tf.ConfigProto(
		gpu_options=gpu_options,
		intra_op_parallelism_threads=num_cores,
		inter_op_parallelism_threads=num_cores, 
		allow_soft_placement=True,
		device_count = {'CPU': num_CPU,'GPU': num_GPU}
	)

	session = tf.Session(config=config)
	K.set_session(session)
	K.set_image_dim_ordering('tf')
