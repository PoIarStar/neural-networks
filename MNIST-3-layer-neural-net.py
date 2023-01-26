import sys, numpy
from keras.datasets import mnist


(x_train, y_train), (x_tests, y_tests) = mnist.load_data()

images, labels = (x_train[:1000].reshape(100, 28*28) / 255, y_train[0:1000])
one_hot_labels = numpy.zeros((len(labels), 10))

for i, j in enumerate(labels):
    one_hot_labels[i][j] = 1
labels = one_hot_labels

test_images = x_tests.reshape(len(x_tests), 28*28) / 255
test_labels = numpy.zeros((len(y_tests), 10))
for i in 
