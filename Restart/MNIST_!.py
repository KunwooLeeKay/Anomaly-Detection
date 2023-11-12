import cv2
print(cv2.__version__)

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./samples/MNIST_data/", one_hot=True)