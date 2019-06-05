import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten


# Training Neural Network Model
def neuralNetwork(inputData, outputClasses): 

    # Convolutions
    conv1 = convLayer(inputData, filterShape = (5,5,32) , dropout=True, dropoutKeepProb = 0.9)
    maxPool1 = maxPool(conv1)
    conv2 = convLayer(maxPool1, filterShape = (5,5,64) , dropout=True, dropoutKeepProb = 0.9)
    maxPool2 = maxPool(conv2)
    conv3 = convLayer(maxPool2, filterShape = (5,5,128) , dropout=True, dropoutKeepProb = 0.9)
    
    convolutionOutput = flatten(conv3)

    # Fully Connected
    fullyConn1 = fullyConnectedLayer(convolutionOutput, outputShape = 1024 , dropout=True, dropoutKeepProb = 0.8)
    fullyConn2 = fullyConnectedLayer(fullyConn1, outputShape = 256 , dropout=True, dropoutKeepProb = 0.8)
    fullyConn3 = fullyConnectedLayer(fullyConn2, outputShape = outputClasses ,relu = False)

    logits = fullyConn3

    return logits


# Prediction Neural Network model (No Dropouts)
def neuralNetworkFull(inputData, outputClasses): 

    # Convolutions
    conv1 = convLayer(inputData, filterShape = (5,5,32) , dropout=True, dropoutKeepProb = 1.0)
    maxPool1 = maxPool(conv1)
    conv2 = convLayer(maxPool1, filterShape = (5,5,64) , dropout=False)
    maxPool2 = maxPool(conv2)
    conv3 = convLayer(maxPool2, filterShape = (5,5,128) , dropout=False)
    
    convolutionOutput = flatten(conv3)
    
    # Fully Connected
    fullyConn1 = fullyConnectedLayer(convolutionOutput, outputShape = 1024 , dropout=True, dropoutKeepProb = 1.0)
    fullyConn2 = fullyConnectedLayer(fullyConn1, outputShape = 256 , dropout=True, dropoutKeepProb = 1.0)
    fullyConn3 = fullyConnectedLayer(fullyConn2, outputShape = outputClasses ,relu = False)

    logits = fullyConn3

    return logits





# Convolution Layer
def convLayer(inputData, filterShape, strides=[1, 1, 1, 1], padding='VALID',  mu = 0 , sigma = 0.1, relu = True, dropout = False, dropoutKeepProb = 0.5):

    inputShape = getTensorShape(inputData)  # Get Input Shape

    # Setup Dropout
    if(dropout):
        inputData = tf.nn.dropout(inputData, dropoutKeepProb)

    # Weights and Biases
    weights = tf.Variable(tf.truncated_normal(shape=(filterShape[0], filterShape[1], inputShape[3], filterShape[2]), mean = mu, stddev = sigma))
    biases = tf.Variable(tf.zeros(filterShape[2]))

    # Convolution Output
    output   = tf.nn.conv2d(inputData, weights, strides=strides, padding=padding) + biases

    # Relu Activation Function
    if(relu):
        output = tf.nn.relu(output)

    return output


# MaxPool Layer
def maxPool(inputData, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'):

    # MaxPool Output
    output = tf.nn.max_pool(inputData, ksize=ksize, strides=strides, padding=padding)

    return output


# Fully Connected Layer
def fullyConnectedLayer(inputData, outputShape, mu = 0 , sigma = 0.1, relu = True, dropout = False, dropoutKeepProb = 0.5):

    inputShape = getTensorShape(inputData)  # Get Input Shape

    # Setup Dropout
    if(dropout):
        inputData = tf.nn.dropout(inputData, dropoutKeepProb)

    # Weights and Biases
    weights = tf.Variable(tf.truncated_normal(shape=(inputShape[1], outputShape), mean = mu, stddev = sigma))
    biases = tf.Variable(tf.zeros(outputShape))
    
    # Convolution Output
    output   = tf.matmul(inputData, weights) + biases

    # Relu Activation Function
    if(relu):
        output = tf.nn.relu(output)

    return output


# Returns Tensor Shape
def getTensorShape(tensor):
    shape = tensor.get_shape().as_list()
    return shape
