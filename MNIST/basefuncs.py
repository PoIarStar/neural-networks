import numpy


def relu(x):
    return x * (x > 0)


def relu2deriv(x):
    return x > 0


def tanh(x):
    return numpy.tanh(x)


def tanh2deriv(output):
    return 1 - (output ** 2)


def softmax(x):
    temp = numpy.exp(x)
    return temp / numpy.sum(temp, axis=1, keepdims=True)
