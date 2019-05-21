import tensorflow as tf
from tensorflow.contrib.layers import flatten


def LeNetModified(inputData, inputShape, outputClasses):    
    
    conv1, conv1Shape = convLayer(inputData, inputShape, filterShape = (5,5,6) )

    maxPool1, maxPool1Shape = maxPool(conv1, conv1Shape)


    conv2, conv2Shape = convLayer(maxPool1, maxPool1Shape, filterShape = (5,5,16) )

    maxPool2, maxPool2Shape = maxPool(conv2, conv2Shape)


    convolutionOutuput = flatten(maxPool2)
    convolutionOutputSize = maxPool2Shape[0] * maxPool2Shape[1] * maxPool2Shape[2]



    fullyConn1, fullyConn1Shape = fullyConnectedLayer(convolutionOutuput, shape = (convolutionOutputSize, 120) )

    fullyConn2, fullyConn2Shape = fullyConnectedLayer(fullyConn1, shape = (fullyConn1Shape, 84) )

    fullyConn3, fullyConn3Shape = fullyConnectedLayer(fullyConn2, shape = (fullyConn2Shape, outputClasses) )

    logits = fullyConn3

    return logits





def convLayer(inputData, inputShape, filterShape, strides=[1, 1, 1, 1], padding='VALID',  mu = 0 , sigma = 0.1, relu = True):

    weights = tf.Variable(tf.truncated_normal(shape=(filterShape[0], filterShape[1], inputShape[2], filterShape[2]), mean = mu, stddev = sigma))
    biases = tf.Variable(tf.zeros(filterShape[2]))
    output   = tf.nn.conv2d(inputData, weights, strides=strides, padding=padding) + biases

    outputHeight = (inputShape[1] - filterShape[0])/strides[1] + 1
    outputWidth = (inputShape[0] - filterShape[1])/strides[2] + 1
    outputDepth = filterShape[2]

    outputShape = (int(outputWidth), int(outputHeight), int(outputDepth))


    if(relu):
        output = tf.nn.relu(output)

    return output, outputShape



def maxPool(inputData, inputShape, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID'):



    output = tf.nn.max_pool(inputData, ksize=ksize, strides=strides, padding=padding)

    outputHeight = (inputShape[1] - ksize[1])/strides[1] + 1
    outputWidth = (inputShape[0] - ksize[2])/strides[2] + 1
    outputDepth = inputShape[2]

    outputShape = (int(outputWidth), int(outputHeight), int(outputDepth))

    return output, outputShape



def fullyConnectedLayer(inputData, shape, mu = 0 , sigma = 0.1, relu = True, dropout = False, dropoutKeepProb = 0.5):

    outputShape = shape[1]

    weights = tf.Variable(tf.truncated_normal(shape=shape, mean = mu, stddev = sigma))
    biases = tf.Variable(tf.zeros(shape[1]))
    output   = tf.matmul(inputData, weights) + biases

    if(relu):
        output = tf.nn.relu(output)

    if(dropout):
        output = tf.nn.dropout(output, dropoutKeepProb)

    return output, outputShape