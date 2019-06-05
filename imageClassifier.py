import numpy as np
import os
import sys
import Plots as plot
import tensorflow as tf
import networks as nn
import matplotlib.pyplot as plt
import cv2


# Define Tensors
classesSize = 43    # Number of classes

x = tf.placeholder(tf.float32,(None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
oneHotY = tf.one_hot(y, classesSize)

logits = nn.neuralNetworkFull(x, classesSize) # No Dropout Model

saver = tf.train.Saver()

# Load Internet Images and Run Model
for i in os.listdir("./InternetImages"):
    image = cv2.imread(("./InternetImages/" + i))   # Load Image
    imageLabel = i.replace('.jpg', '')  # Load Label

    


    # Preprocess image
    imageArray = np.zeros(dtype = np.float32, shape = (1,image.shape[0], image.shape[1], 1))

    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.uint8) # To Grayscale
    histogramEqualized= cv2.equalizeHist(grayscale) # Histogram Equalization
    normalized = ((histogramEqualized - 128.0) / 128.0).astype(np.float32)  # Normalization

    imageArray[0,:,:,0] = normalized    # To Tensor Format


    # Run Session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer()) # Initialize Global Variables
        saver.restore(sess, tf.train.latest_checkpoint('.'))    # Load Trained Model

        result = sess.run(logits, feed_dict={x: imageArray})    # Run Model

        # Calculate Softmax Probabilities
        result = np.array(result[0], dtype=np.float64)
        expVals = np.exp(result)
        softMaxProbs = expVals / np.sum(expVals)
        maxValueIdx = np.argmax(softMaxProbs)

        # Plot Results
        yVal = softMaxProbs*100
        theTitle = "Image Label: {}, Prediction: {}".format(imageLabel, maxValueIdx)
        plot.barPlot2(np.arange(0,classesSize,1), yVal, xLabel='Classes', setXAxis= (-1,43),
                yLabel='Logit Values [%]', fileName=("Internet_" + imageLabel), save=True, show=True, title = theTitle)
        






