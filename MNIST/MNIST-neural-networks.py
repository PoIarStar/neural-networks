from basefuncs import tanh
from PIL import Image
import numpy
from json import load


with open('weights.json') as file:
    weights = load(file)


def image_to_list(image):
    with Image.open(image) as picture:
        picture = picture.resize((28, 28)).convert('L')
        picture.show()
        picture = list(picture.getdata())
        return picture


def inline_network(image):
    image = numpy.array(image_to_list(image))
    layer_1 = tanh(numpy.dot(image, weights['inline']['weights_0_1']))
    layer_2 = list(numpy.dot(layer_1, weights['inline']['weights_1_2']))
    return layer_2.index(max(layer_2))


'''def convolutional_network(image):
    image = numpy.array(image_to_list(image))'''
