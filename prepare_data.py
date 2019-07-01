import os
from os import path
from urllib import urlretrieve
import tarfile
import cPickle as pickle
import numpy as np


def maybe_download(url, filename):
    filepath = path.join('./data', filename)
    if not path.exists(filepath):
        urlretrieve(url, path.join('./data', filename))
        tar = tarfile.open(filepath, 'r:gz')
        tar.extractall('./data')
        tar.close()
    pass


def load_data(filename, as_list=False):
    file_path = path.join('./data/cifar-10-batches-py/', filename)
    f = open(file_path, 'r')
    dic = pickle.load(f)
    f.close()
    data = dic['data']
    labels = dic['labels']
    if as_list:
        return data, labels
    else:
        return np.array(data), np.array(labels)


def load_train():
    filenames = ['data_batch_{}'.format(n) for n in range(1, 6)]
    data = []
    labels = []
    for f in filenames:
        d, l = load_data(f, as_list=True)
        data.extend(d)
        labels.extend(l)

    return np.array(data), np.array(labels)


def predict(model, x_test, y_test):
    predict_classes = model.predict_classes(x_test)
    accuracy = [x == y for (x, y) in zip(predict_classes, y_test)]
    acc_rate = sum(i for i in accuracy if i) / float(len(y_test)) * 100
    print('accuracy:{}'.format(acc_rate))


if __name__ == '__main__':
    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    filename = url.split('/')[-1]
    maybe_download(url, filename)
