import sys
import numpy
from keras.datasets import mnist


(x_train, y_train), (x_tests, y_tests) = mnist.load_data()

images, labels = (x_train[:1000].reshape(1000, 28*28) / 255, y_train[0:1000])
one_hot_labels = numpy.zeros((len(labels), 10))

for i, j in enumerate(labels):
    one_hot_labels[i][j] = 1
labels = one_hot_labels

test_images = x_tests.reshape(len(x_tests), 28*28) / 255
test_labels = numpy.zeros((len(y_tests), 10))
for i, j in enumerate(y_tests):
    test_labels[i][j] = 1
numpy.random.seed(1)
relu = lambda x: (x >= 0) * x
relu2deriv = lambda x: x >= 0
alpha, iterations, hidden_size, pixels_per_image, num_labels = (0.005, 350, 40, 784, 10)
weights_0_1 = 0.2 * numpy.random.random((pixels_per_image, hidden_size)) - 0.1
weights_1_2 = 0.2 * numpy.random.random((hidden_size, num_labels)) - 0.1

for j in range(iterations):
    error, correct_cnt = (0.0, 0)
    for i in range(len(images)):
        layer_0 = images[i:i+1]
        layer_1 = relu(numpy.dot(layer_0, weights_0_1))
        dropout_mask = numpy.random.randint(2, size=layer_1.shape)

        layer_1 *= dropout_mask * 2
        layer_2 = numpy.dot(layer_1, weights_1_2)
        error += numpy.sum((labels[i:i+1] - layer_2) ** 2)
        correct_cnt += int(numpy.argmax(layer_2) == numpy.argmax(labels[i:i+1]))
        layer_2_delta = (labels[i:i+1] - layer_2)
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
        layer_1_delta *= dropout_mask
        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

    if j % 10 == 0:
        test_error = 0.0
        test_correct_cnt = 0
        for i in range(len(test_images)):
            layer_0 = test_images[i:i + 1]
            layer_1 = relu(numpy.dot(layer_0, weights_0_1))
            layer_2 = numpy.dot(layer_1, weights_1_2)
            test_error += numpy.sum((test_labels[i:i + 1] - layer_2) ** 2)
            test_correct_cnt += int(numpy.argmax(layer_2) == numpy.argmax(test_labels[i:i + 1]))
        sys.stdout.write("\nI:" + str(j) + " Test-Err:" + str(test_error / float(len(test_images)))[0:5] +
                         " Test-Acc:" + str(test_correct_cnt / float(len(test_images))) +
                         " Train-Err:" + str(error / float(len(images)))[0:5] +
                         " Train-Acc:" + str(correct_cnt / float(len(images))))
