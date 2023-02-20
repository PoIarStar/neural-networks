import sys
import numpy
from keras.datasets import mnist
import json
from basefuncs import tanh, tanh2deriv, softmax


if __name__ == '__main__':
    (x_train, y_train), (x_tests, y_tests) = mnist.load_data()

    images, labels = (x_train[:1000].reshape(1000, 28 * 28) / 255, y_train[0:1000])
    one_hot_labels = numpy.zeros((len(labels), 10))

    for i, j in enumerate(labels):
        one_hot_labels[i][j] = 1
    labels = one_hot_labels

    test_images = x_tests.reshape(len(x_tests), 28 * 28) / 255
    test_labels = numpy.zeros((len(y_tests), 10))
    for i, j in enumerate(y_tests):
        test_labels[i][j] = 1
    numpy.random.seed(1)
    alpha, iterations, hidden_size, batch_size, pixels_per_image, num_labels = (10, 231, 100, 100, 784, 10)
    weights_0_1 = 0.02 * numpy.random.random((pixels_per_image, hidden_size)) - 0.01
    weights_1_2 = 0.2 * numpy.random.random((hidden_size, num_labels)) - 0.1

    for j in range(iterations):
        error, correct_cnt = (0.0, 0)
        for i in range(int(len(images) / batch_size)):
            batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))
            layer_0 = images[batch_start:batch_end]
            layer_1 = tanh(numpy.dot(layer_0, weights_0_1))
            dropout_mask = numpy.random.randint(2, size=layer_1.shape)
            layer_1 *= dropout_mask * 2
            layer_2 = softmax(numpy.dot(layer_1, weights_1_2))
            error += numpy.sum((labels[i:i + 1] - layer_2) ** 2)
            for k in range(batch_size):
                correct_cnt += int(numpy.argmax(layer_2[k:k + 1]) ==
                                   numpy.argmax(labels[batch_start + k:batch_start + k + 1]))
            layer_2_delta = (labels[batch_start:batch_end] - layer_2) / (batch_size * layer_2.shape[0])
            layer_1_delta = layer_2_delta.dot(weights_1_2.T) * tanh2deriv(layer_1)
            layer_1_delta *= dropout_mask
            weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
            weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)

        if j % 1 == 0:
            test_error = 0.0
            test_correct_cnt = 0
            for i in range(len(test_images)):
                layer_0 = test_images[i:i + 1]
                layer_1 = tanh(numpy.dot(layer_0, weights_0_1))
                layer_2 = numpy.dot(layer_1, weights_1_2)
                test_error += numpy.sum((test_labels[i:i + 1] - layer_2) ** 2)
                test_correct_cnt += int(numpy.argmax(layer_2) == numpy.argmax(test_labels[i:i + 1]))
            sys.stdout.write("\nI:" + str(j) + " Test-Err:" + str(test_error / float(len(test_images)))[0:5] +
                             " Test-Acc:" + str(test_correct_cnt / len(test_images)) +
                             " Train-Err:" + str(error / len(images))[0:5] +
                             " Train-Acc:" + str(correct_cnt / len(images)))

            with open('MNIST\\weights.json') as file:
                weights = json.load(file)

            if test_correct_cnt / len(test_images) > weights['inline']['accuracy']:
                weights['inline']['weights_0_1'] = weights_0_1.tolist()
                weights['inline']['weights_1_2'] = weights_1_2.tolist()
                weights['inline']['accuracy'] = test_correct_cnt / len(test_images)

            with open('MNIST\\weights.json', 'w') as file:
                json.dump(weights, file)
