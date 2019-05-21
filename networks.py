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
        print('Dropout')

    return output, outputShape

def LeNet(x, image_shape, n_classes,  mu = 0 , sigma = 0.1):    
    
    # SOLUTION: Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, image_shape[2], 6), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # SOLUTION: Activation.
    conv1 = tf.nn.relu(conv1)

    # SOLUTION: Pooling. Input = 28x28x6. Output = 14x14x6.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Layer 2: Convolutional. Output = 10x10x16.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 6, 16), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # SOLUTION: Activation.
    conv2 = tf.nn.relu(conv2)

    # SOLUTION: Pooling. Input = 10x10x16. Output = 5x5x16.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # SOLUTION: Flatten. Input = 5x5x16. Output = 400.
    fc0   = flatten(conv2)
    
    # SOLUTION: Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # SOLUTION: Activation.
    fc1    = tf.nn.relu(fc1)

    # SOLUTION: Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # SOLUTION: Activation.
    fc2    = tf.nn.relu(fc2)

    # SOLUTION: Layer 5: Fully Connected. Input = 84. Output = n_classes.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, n_classes), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(n_classes))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits