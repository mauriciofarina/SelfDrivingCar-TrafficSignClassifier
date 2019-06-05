import numpy as np


np.warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import sys

import Plots as plot

import tensorflow as tf

import networks as nn
import matplotlib.pyplot as plt

import cv2




def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(8,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
    plt.show()


def getMyFeatureMaps(imageArray, convLayers):

    outputFeatureMap(imageArray, convLayers[0])
    outputFeatureMap(imageArray, convLayers[1])



classesSize = 43

x = tf.placeholder(tf.float32,(None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
oneHotY = tf.one_hot(y, classesSize)


logits, convLayers = nn.neuralNetworkFull(x, classesSize)



saver = tf.train.Saver()


for i in os.listdir("./InternetImages"):
    image = cv2.imread(("./InternetImages/" + i))
    imageLabel = i.replace('.jpg', '')

    
    imageArray = np.zeros(dtype = np.float32, shape = (1,image.shape[0], image.shape[1], 1))

    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    histogramEqualized= cv2.equalizeHist(grayscale)
    normalized = ((histogramEqualized - 128.0) / 128.0).astype(np.float32)


    imageArray[0,:,:,0] = normalized





    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        result = sess.run(logits, feed_dict={x: imageArray})

        #getMyFeatureMaps(imageArray, convLayers)

        result = np.array(result[0], dtype=np.float64)

        expVals = np.exp(result)

        softMaxProbs = expVals / np.sum(expVals)
    
        maxValueIdx = np.argmax(softMaxProbs)

        yVal = softMaxProbs*100
        theTitle = "Image Label: {}, Prediction: {}".format(imageLabel, maxValueIdx)
        plot.barPlot2(np.arange(0,classesSize,1), yVal, xLabel='Classes', setXAxis= (-1,43),
                yLabel='Logit Values [%]', fileName='Test', save=False, show=True, title = theTitle)
        






