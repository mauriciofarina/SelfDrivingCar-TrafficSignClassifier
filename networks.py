import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten



def neuralNetwork(inputData, outputClasses): 

    conv1 = convLayer(inputData, filterShape = (5,5,32) , dropout=True, dropoutKeepProb = 0.9)
    maxPool1 = maxPool(conv1)
    conv2 = convLayer(maxPool1, filterShape = (5,5,64) , dropout=False, dropoutKeepProb = 0.9)
    maxPool2 = maxPool(conv2)
    conv3 = convLayer(maxPool2, filterShape = (5,5,128) , dropout=False, dropoutKeepProb = 0.9)
    
    convolutionOutput = flatten(conv3)
    
    fullyConn1 = fullyConnectedLayer(convolutionOutput, outputShape = 1024 , dropout=True, dropoutKeepProb = 0.5)

    fullyConn2 = fullyConnectedLayer(fullyConn1, outputShape = 256 , dropout=False, dropoutKeepProb = 0.5)

    fullyConn3 = fullyConnectedLayer(fullyConn2, outputShape = outputClasses ,relu = False)

    logits = fullyConn3

    
    return logits


def neuralNetworkFull(inputData, outputClasses): 

    conv1 = convLayer(inputData, filterShape = (5,5,32) , dropout=False, dropoutKeepProb = 0.9)
    maxPool1 = maxPool(conv1)
    conv2 = convLayer(maxPool1, filterShape = (5,5,64) , dropout=False, dropoutKeepProb = 0.9)
    maxPool2 = maxPool(conv2)
    conv3 = convLayer(maxPool2, filterShape = (5,5,128) , dropout=False, dropoutKeepProb = 0.9)
    
    convolutionOutput = flatten(conv3)
    
    fullyConn1 = fullyConnectedLayer(convolutionOutput, outputShape = 1024 , dropout=False, dropoutKeepProb = 0.5)

    fullyConn2 = fullyConnectedLayer(fullyConn1, outputShape = 256 , dropout=False, dropoutKeepProb = 0.5)

    fullyConn3 = fullyConnectedLayer(fullyConn2, outputShape = outputClasses ,relu = False)

    logits = fullyConn3

    
    return logits, (conv1, conv2)






def convLayer(inputData, filterShape, strides=[1, 1, 1, 1], padding='VALID',  mu = 0 , sigma = 0.1, relu = True, dropout = False, dropoutKeepProb = 0.5):

    inputShape = getTensorShape(inputData)

    if(dropout):
        inputData = tf.nn.dropout(inputData, dropoutKeepProb)

    weights = tf.Variable(tf.truncated_normal(shape=(filterShape[0], filterShape[1], inputShape[3], filterShape[2]), mean = mu, stddev = sigma))
    biases = tf.Variable(tf.zeros(filterShape[2]))
    output   = tf.nn.conv2d(inputData, weights, strides=strides, padding=padding) + biases

    if(relu):
        output = tf.nn.relu(output)

    return output



def maxPool(inputData, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'):

    output = tf.nn.max_pool(inputData, ksize=ksize, strides=strides, padding=padding)

    return output



def fullyConnectedLayer(inputData, outputShape, mu = 0 , sigma = 0.1, relu = True, dropout = False, dropoutKeepProb = 0.5):

    inputShape = getTensorShape(inputData)

    if(dropout):
        inputData = tf.nn.dropout(inputData, dropoutKeepProb)

    weights = tf.Variable(tf.truncated_normal(shape=(inputShape[1], outputShape), mean = mu, stddev = sigma))
    biases = tf.Variable(tf.zeros(outputShape))
    output   = tf.matmul(inputData, weights) + biases

    
    if(relu):
        output = tf.nn.relu(output)

    return output


def getTensorShape(tensor):
    shape = tensor.get_shape().as_list()
    return shape
