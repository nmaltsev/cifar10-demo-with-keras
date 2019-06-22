
import cPickle as pickle
import numpy as np
from PIL import Image

# https://raw.githubusercontent.com/dsanno/chainer-cifar/master/src/dataset.py

def save_image(image, path, normalize=True):
    if normalize:
        max_value = np.max(np.abs(image), axis=1).reshape((100, 1))
        image = image / max_value * 127
    image = (image + 128).clip(0, 255).astype(np.uint8)
    image = image.reshape((10, 10, 3, 32, 32))
    image = np.pad(image, ((0, 0), (0, 0), (0, 0), (2, 2), (2, 2)), mode='constant', constant_values=0)
    image = image.transpose((0, 3, 1, 4, 2)).reshape((360, 360, 3))
    Image.fromarray(image).save(path)


class Normalization:
    @staticmethod
    def subtractMean(raw_train_x, raw_test_x):
        mean = np.mean(raw_train_x)
        train_x = raw_train_x - mean
        test_x = raw_test_x - mean    
        return (train_x, test_x)
    
    @staticmethod    
    def ZCAWhitening(raw_train_x, raw_test_x):
        zca, mean = Normalization._calc_zca(raw_train_x)
        train_x = np.dot(raw_train_x - mean, zca.T)
        test_x = np.dot(raw_test_x - mean, zca.T)
        return (train_x, test_x)

    @staticmethod    
    def _calc_zca(x):
        n = x.shape[0]

        mean = np.mean(x, axis=0)
        x = x - mean

        c = np.dot(x.T, x)
        u, lam, v = np.linalg.svd(c)

        eps = 0
        sqlam = np.sqrt(lam + eps)
        uzca = np.dot(u / sqlam[np.newaxis, :], u.T)
        return uzca, mean

    @staticmethod
    def _calc_mean(x):
        return x.reshape((-1, 3, 32 * 32)).mean(axis=(0, 2))

    @staticmethod
    def _calc_std(x):
        return x.reshape((-1, 3, 32 * 32)).std(axis=(0, 2))

    @staticmethod
    def _normalize_dataset(x, mean, std=None):
        shape = x.shape
        x = x.reshape((-1, 3)) - mean
        if std is not None:
            x /= std
        return x.reshape(shape)
    @staticmethod
    def _normalize_contrast(x):
        mean = np.mean(x, axis=1).reshape((-1, 1))
        std = np.std(x, axis=1).reshape((-1, 1))

        return (x - mean) / std

    @staticmethod
    def contrastWithZCA(raw_train_x, raw_test_x):
        # contrast normalization and ZCA whitening
        train_x = Normalization._normalize_contrast(raw_train_x)
        test_x = Normalization._normalize_contrast(raw_test_x)
        zca, mean = Normalization._calc_zca(train_x)
        train_x = np.dot(train_x - mean, zca.T)
        test_x = np.dot(test_x- mean, zca.T)
        return (train_x, test_x)

def dumpDatasetWithNormalization(raw_train_x, raw_test_x, normalizationMethod, dumpPath_s, imagePath_s):
    normalizedData = normalizationMethod(raw_train_x, raw_test_x)
    with open(dumpPath_s, 'wb') as f:
        pickle.dump({
            'train': normalizedData[0], 
            'test': normalizedData[1]
            }, f, pickle.HIGHEST_PROTOCOL)
    save_image(
        normalizedData[0][:100, ], 
        imagePath_s
    )

def dumpMatrix(matrix, dumpPath_s):
    with open(dumpPath_s, 'wb') as f:
        pickle.dump(matrix, f, pickle.HIGHEST_PROTOCOL)