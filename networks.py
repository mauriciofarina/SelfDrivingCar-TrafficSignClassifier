import tensorflow as tf
import numpy as np
from tensorflow.contrib.layers import flatten



def LeNetModified(inputData, outputClasses):    

    '''
    conv1 = convLayer(inputData, filterShape = (5,5,6) )
    maxPool1 = maxPool(conv1)
    

    conv2A = convLayer(maxPool1, filterShape = (5,5,16) )
    maxPool2A = maxPool(conv2A)
    #conv3A = convLayer(maxPool2A, filterShape = (3,3,40) )
    #conv4A = convLayer(conv3A, filterShape = (1,1,20) )
    conv3A = convLayer(maxPool2A, filterShape = (1,1,20) )



    #conv2B = convLayer(maxPool1, filterShape = (3,3,15) )
    #maxPool2B = maxPool(conv2B)
    #conv3B = convLayer(maxPool2B, filterShape = (3,3,10) )
    #conv4B = convLayer(conv3B, filterShape = (1,1,10) )

    '''


    conv1A = convLayer(inputData, filterShape = (5,5,6) )
    maxPool1A = maxPool(conv1A)
    conv2A = convLayer(maxPool1A, filterShape = (5,5,16) )
    maxPool2A = maxPool(conv2A)
    
    conv1B = convLayer(inputData, filterShape = (3,3,6) )
    maxPool1B = maxPool(conv1B)
    conv2B = convLayer(maxPool1B, filterShape = (3,3,24) )
    maxPool2B = maxPool(conv2B)

    
    conv1C = convLayer(inputData, filterShape = (32,32,10))


    convOutA = flatten(maxPool2A)
    convOutB = flatten(maxPool2B)
    convOutC = flatten(conv1C)

    convolutionOutput = tf.concat([convOutA, convOutB, convOutC], 1)

    

    fullyConn1 = fullyConnectedLayer(convolutionOutput, outputShape = 160 , dropout=False)

    fullyConn2 = fullyConnectedLayer(fullyConn1, outputShape = 120 , dropout=False)

    fullyConn3 = fullyConnectedLayer(fullyConn2, outputShape = 84 , dropout=False)

    #fullyConn4 = fullyConnectedLayer(fullyConn3, outputShape = 40 , dropout=False)

    fullyConn4 = fullyConnectedLayer(fullyConn3, outputShape = outputClasses, relu = False)

    logits = fullyConn4

    
    return logits



'''
def LeNet(inputData, outputClasses):    
    
    conv1 = convLayer(inputData, filterShape = (5,5,6) )

    maxPool1 = maxPool(conv1)


    conv2 = convLayer(maxPool1, filterShape = (5,5,16) )

    maxPool2 = maxPool(conv2)


    convolutionOutput = flatten(maxPool2)


    fullyConn1 = fullyConnectedLayer(convolutionOutput, outputShape = 120 , dropout=True)

    fullyConn2 = fullyConnectedLayer(fullyConn1, outputShape = 100 , dropout=True)

    fullyConn3 = fullyConnectedLayer(fullyConn2, outputShape = 80 , dropout=True)

    fullyConn4 = fullyConnectedLayer(fullyConn3, outputShape = outputClasses, relu = False)

    logits = fullyConn4

    return logits

'''



def convLayer(inputData, filterShape, strides=[1, 1, 1, 1], padding='VALID',  mu = 0 , sigma = 0.1, relu = True):

    inputShape = getTensorShape(inputData)

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

    weights = tf.Variable(tf.truncated_normal(shape=(inputShape[1], outputShape), mean = mu, stddev = sigma))
    biases = tf.Variable(tf.zeros(outputShape))
    output   = tf.matmul(inputData, weights) + biases

    if(relu):
        output = tf.nn.relu(output)

    if(dropout):
        output = tf.nn.dropout(output, dropoutKeepProb)

    return output


def getTensorShape(tensor):
    shape = tensor.get_shape().as_list()
    return shape
