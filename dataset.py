import os
import six
import cPickle as pickle
import numpy as np
from keras.utils import np_utils

from libs.dataset_utils import dumpDatasetWithNormalization, Normalization


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

def main2(dataset_path, destination_path):
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

# @param {Number} slave_n - number of slaves
def prepare_dataset_chunks(slaves_n, dataset_path_s, destination_path_s):
    print('Slaves', slaves_n)
    
    # print("Save dataset for ", nb_slaves, " slaves")
    # batch_size = len(train)//nb_slaves
    raw_train_x, raw_train_y, raw_test_x, raw_test_y = load(dataset_path_s)
    
    # Create destination path if it does not exist
    if not os.path.exists(destination_path_s):
        os.mkdir(destination_path_s)
        
    

if __name__ == '__main__':
    dataset_path = '/root/tfplayground/datasets/cifar-10-batches-py'
    # dataset_path = '/media/cluster_files/dev/cifar/cifar-10-batches-py'
    main2(dataset_path, 'data')
