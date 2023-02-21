from basefuncs import tanh
from PIL import Image
import numpy
from json import load


kernel_rows, kernel_cols = 3, 3


with open('weights.json') as file:
    weights = load(file)


def get_image_section(image, left, width, top, height):
    return numpy.array([[[i[left:left+width] for i in image[top:top + height]]]])


def image_to_list(image):
    with Image.open(image) as picture:
        picture = picture.resize((28, 28)).convert('L')
        picture = list(picture.getdata())
        return picture


def inline_network(image):
    image = numpy.array(image_to_list(image))
    layer_1 = tanh(numpy.dot(image, weights['inline']['weights_0_1']))
    layer_2 = list(numpy.dot(layer_1, weights['inline']['weights_1_2']))
    return layer_2.index(max(layer_2))


def convolutional_network(image):
    image = numpy.array([image_to_list(image)]).reshape(28, 28)
    sects = list()
    for row_start in range(image.shape[0] - kernel_rows):
        for col_start in range(image.shape[1] - kernel_cols):
            sect = get_image_section(image,
                                     row_start,
                                     kernel_rows,
                                     col_start,
                                     kernel_cols)
            sects.append(sect)

    expanded_input = numpy.concatenate(sects, axis=1)
    es = expanded_input.shape
    flattened_input = expanded_input.reshape(es[0] * es[1], -1)
    kernel_output = flattened_input.dot(weights['convolutional']['kernels'])
    layer_1 = tanh(kernel_output.reshape(es[0], -1))
    layer_2 = numpy.dot(layer_1, weights['convolutional']['weights_1_2'])[0]
    return list(layer_2.index(max(layer_2)))
