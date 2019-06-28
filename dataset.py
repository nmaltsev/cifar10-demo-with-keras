import sys
import os
import six
import cPickle as pickle
import numpy as np
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('tf')
from libs.dataset_utils import dumpDatasetWithNormalization, Normalization, dumpMatrix


train_files = ['data_batch_{}'.format(i+1) for i in six.moves.range(5)]
test_files = ['test_batch']


def load_file(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data['data'].astype(np.float32), np.asarray(data['labels'], dtype=np.int32)


def load(data_dir):
    train_data = [load_file(os.path.join(data_dir, file_name)) for file_name in train_files]
    images, labels = zip(*train_data)
    train_images = np.concatenate(images)
    train_labels = np.concatenate(labels)
    ## TODO refactor code
    test_data = [load_file(os.path.join(data_dir, file_name)) for file_name in test_files]
    images, labels = zip(*test_data)
    test_images = np.concatenate(images)
    test_labels = np.concatenate(labels)

    return train_images, train_labels, test_images, test_labels

def restoreDataset():
    # path_s = 'data/image_norm_zca.pkl'
    path_s = 'data/image.pkl'
    with open(path_s, 'rb') as f:
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

    return train_x, train_y, valid_x, valid_y, test_x, test_y

def prepare_dataset(dataset_path, destination_path):
    output_path = destination_path
    raw_train_x, raw_train_y, raw_test_x, raw_test_y = load(dataset_path)

    # save labels
    with open(os.path.join(output_path, 'label.pkl'), 'wb') as f:
        pickle.dump({
            'train': raw_train_y, 
            'test': raw_test_y
        }, f, pickle.HIGHEST_PROTOCOL)

    dumpDatasetWithNormalization(
        raw_train_x, 
        raw_test_x, 
        Normalization.subtractMean, 
        os.path.join(output_path, 'image.pkl'),
        os.path.join(output_path, 'sample.png')
    )
    # dumpDatasetWithNormalization(
    #     raw_train_x, 
    #     raw_test_x, 
    #     Normalization.ZCAWhitening, 
    #     os.path.join(output_path, 'image_zca.pkl'),
    #     os.path.join(output_path, 'sample_zca.png')
    # )
    # dumpDatasetWithNormalization(
    #     raw_train_x, 
    #     raw_test_x, 
    #     Normalization.contrastWithZCA, 
    #     os.path.join(output_path, 'image_norm_zca.pkl'),
    #     os.path.join(output_path, 'sample_zca.png')
    # )

def parseInt(s):
    try:
        val = int(s)
    except:
        val = None
    finally:
        return val 

def splitOnParts(length, numberOfChunks):
    chunkLen = length // numberOfChunks if length % numberOfChunks == 0 else (length // numberOfChunks + 1)
    pos = 0
    n = 0
    while pos < length:
        yield (n, pos, pos + chunkLen if pos + chunkLen < length else length)
        pos += chunkLen
        n += 1


# @param {Number} slave_n - number of slaves
def prepare_dataset_chunks(slaves_n, dataset_path_s, destination_path_s):
    raw_train_x, raw_train_y, raw_test_x, raw_test_y = load(dataset_path_s)
    
    # Create destination path if it does not exist
    if not os.path.exists(destination_path_s):
        os.mkdir(destination_path_s)

    # Normalize data
    x_norm = Normalization.subtractMean(raw_train_x, raw_test_x)

    # x_norm[0] is a train_x
    # x_norm[1] is a test_x

    for step in splitOnParts(len(x_norm[0]), slaves_n):
        train_x = x_norm[0][step[1] : step[2],]
        train_y = raw_train_y[step[1] : step[2]]
        print('step', step, step[2] - step[1])
        print(train_x.shape, train_y.shape)
        dumpMatrix(
            (train_x, train_y),
            'data/train_chunk_{}.pkl'.format(step[0]) # TODO use destination_path_s!!
        )
    
    dumpMatrix(
        (
            x_norm[1], 
            raw_test_y
        ),
        'data/test_chunk.pkl'
    )

def restoreDatasetChunk(chunk_n):
    # TODO refactor this code!
    
    # ~ chankDataPath_s = 'data/train_chunk_{}.pkl'.format(chunk_n)
    chankDataPath_s = '/media/cluster_files/dev/repo/cifar10-demo-with-keras/data/train_chunk_{}.pkl'.format(chunk_n)
    
    ### TODO I need train_x, train_y!
    with open(chankDataPath_s, 'rb') as f:
        (train_chunk_x, train_chunk_y) = pickle.load(f)
        index = np.random.permutation(len(train_chunk_x))
        print('Index shape', index.shape)
        train_index = index[:-500]
        # valid_index = index[-500:]
        train_x = train_chunk_x[train_index].reshape((-1, 3, 32, 32)) # [1..40000]
        # valid_x = train_chunk_x[valid_index].reshape((-1, 3, 32, 32)) # [45000...50000]
        train_y = train_chunk_y[train_index]
        train_y = np_utils.to_categorical(train_y)
        ### test_x in data/test_chunk.pkl[0]
        # test_x = images['test'].reshape((-1, 3, 32, 32))

    # with open('data/label.pkl', 'rb') as f:
    #     labels = pickle.load(f)
    #     train_y = labels['train'][train_index]
    #     valid_y = labels['train'][valid_index]
    #     valid_y = np_utils.to_categorical(valid_y)
    #     test_y = labels['test']

    return train_x, train_y

# for cluster training
def restoreTestDataset():
    # ~ testPath_s = 'data/test_chunk.pkl'
    testPath_s = '/media/cluster_files/dev/repo/cifar10-demo-with-keras/data/test_chunk.pkl'
    
    with open(testPath_s, 'rb') as f:
        (test_x, test_y) = pickle.load(f)
        test_x = test_x.reshape((-1, 3, 32, 32))
        test_y = np_utils.to_categorical(test_y)
    return test_x, test_y

if __name__ == '__main__':
    if len(sys.argv) == 3 :
        workers_n = parseInt(sys.argv[2]) or 1
    else:
        workers_n = 1
    
    # dataset_path = '/root/tfplayground/datasets/cifar-10-batches-py'
    # dataset_path = '/media/cluster_files/dev/cifar/cifar-10-batches-py'

    dataset_path = sys.argv[1]
    destination_path = 'data'
    
    if workers_n == 1:
        prepare_dataset(dataset_path, destination_path)
    else:
        prepare_dataset_chunks(workers_n, dataset_path, destination_path)
    
