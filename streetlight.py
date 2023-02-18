import numpy

numpy.random.seed(1)


def relu(x):
    return (x > 0) * x


def relu2deriv(output):
    return output > 0


def neuronet(input):
    layer_1 = relu(numpy.dot(input, weights_0_1))
    layer_2 = numpy.dot(layer_1, weights_1_2)
    return round(layer_2[0])


streetlights = numpy.array([[1, 0, 1],
                            [0, 1, 1],
                            [0, 0, 1],
                            [1, 1, 1]])

walk_vs_stop = numpy.array([[0, 1, 0, 1]]).T

alpha = 0.2
hidden_size = 4

weights_0_1 = 2 * numpy.random.random((3, hidden_size)) - 1
weights_1_2 = 2 * numpy.random.random((hidden_size, 1)) - 1

for iteration in range(60):
    layer_2_error = 0
    for i in range(len(streetlights)):
        layer_0 = streetlights[i:i + 1]
        layer_1 = relu(numpy.dot(layer_0, weights_0_1))
        layer_2 = numpy.dot(layer_1, weights_1_2)
        layer_2_error += numpy.sum((layer_2 - walk_vs_stop[i:i + 1]) ** 2)
        layer_2_delta = (walk_vs_stop[i:i + 1] - layer_2)
        layer_1_delta = layer_2_delta.dot(weights_1_2.T) * relu2deriv(layer_1)
        weights_1_2 += alpha * layer_1.T.dot(layer_2_delta)
        weights_0_1 += alpha * layer_0.T.dot(layer_1_delta)
    if iteration % 10 == 9:
        print("Error:" + str(layer_2_error))

print(neuronet([0, 1, 0]))
