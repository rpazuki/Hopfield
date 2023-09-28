import os
import gzip
import numpy as np


def load_mnist(path, kind='train'):    

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

def get_images(size=9):
    X_train, y_train = load_mnist('./data/fashion', kind='train')
    return [X_train[i+9].copy().reshape((28, 28)) for i in range(size)]
    
def get_binary_image(size=9, threshold=80):
    images = get_images(size)
    return [np.where(img > threshold, 0, 1) for img in images]
    