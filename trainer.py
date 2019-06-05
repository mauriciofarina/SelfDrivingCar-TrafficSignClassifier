import numpy as np
import time
import os
import sys
import pickle
import Plots as plot
import tensorflow as tf
from sklearn.utils import shuffle
import networks as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.colors as clr
import cv2


PREVIEW = False  # Plot Preview On/Off

# Default Hyperparameters
EPOCHS = 20     # Number of Epochs
BATCH_SIZE = 128    # Image Batch Size
LEARNING_RATE = 0.001   # Learning Rate

# Run Model on Test Dataset Flag
TEST_NOW = False


# Script Args
for index, value in enumerate(sys.argv):
    
    if value == "-e":   # Set Epoch Value
        EPOCHS = int(sys.argv[index+1])
    
    elif value == "-b": # Set Batch Size
        BATCH_SIZE = int(sys.argv[index+1])

    elif value == "-l": # Set Learning Rate
        LEARNING_RATE = float(sys.argv[index+1])

    elif value == "-t": # Run Model on Test Dataset
        TEST_NOW = True



# Dataset Files Path
trainingFile = '../data/train.p'
validationFile = '../data/valid.p'
testingFile = '../data/test.p'

# Open Dataset Files
with open(trainingFile, mode='rb') as f:
    train = pickle.load(f)
with open(validationFile, mode='rb') as f:
    valid = pickle.load(f)
with open(testingFile, mode='rb') as f:
    test = pickle.load(f)

# Load Dataset Values
xTrain, yTrain = train['features'], train['labels']
xValid, yValid = valid['features'], valid['labels']
xTest, yTest = test['features'], test['labels']



# Number of training examples
trainSize = np.shape(xTrain)[0]

# Number of validation examples
validationSize = np.shape(xValid)[0]

# Number of testing examples.
testSize = np.shape(xTest)[0]

# What's the shape of an traffic sign image?
imageShape = (np.shape(xTrain)[1], np.shape(xTrain)[2], np.shape(xTrain)[3])

# List of Classes
classesList = np.unique(yTrain)

# How many unique classes/labels there are in the dataset.
classesSize = np.shape(classesList)[0]


# Plot Items Per Dataset Group
xVal = ['Training', 'Validation', 'Testing']
yVal = [trainSize, validationSize, testSize]
plot.barPlot(xVal, yVal, xLabel='Dataset Groups',
             yLabel='Number of Images', fileName='datasetGroups', save=True, show=PREVIEW)


# Plot Total of Items per Class in Training Dataset
yVal = np.zeros(shape=(classesSize))

for i in range(0, classesSize, 1):
    yVal[i] = np.count_nonzero(yTrain == i)
    
histTrain = plot.histogramPlot(yVal, bins=classesSize, xLabel='Classes',
                   yLabel='Number of Images', fileName='datasetHistTrain', save=False, density=True, show=PREVIEW)

plot.barPlot2(np.arange(0,classesSize,1), yVal, xLabel='Classes', setXAxis= (-1,43),
             yLabel='Number of Images', fileName='datasetHistTrain', save=True, show=PREVIEW)


# Plot Total of Items per Class in Validation Dataset
yVal = np.zeros(shape=(classesSize))

for i in range(0, classesSize, 1):
    yVal[i] = np.count_nonzero(yValid == i)

histValid = plot.histogramPlot(yVal, bins=classesSize, xLabel='Classes',
                   yLabel='Number of Images', fileName='datasetHistValid', save=False, density=True, show=PREVIEW)

plot.barPlot2(np.arange(0,classesSize,1), yVal, xLabel='Classes', setXAxis= (-1,43),
             yLabel='Number of Images', fileName='datasetHistValid', save=True, show=PREVIEW)


# Plot Total of Items per Class in Testing Dataset
yVal = np.zeros(shape=(classesSize))

for i in range(0, classesSize, 1):
    yVal[i] = np.count_nonzero(yTest == i)

histTest = plot.histogramPlot(yVal, bins=classesSize, xLabel='Classes',
                   yLabel='Number of Images', fileName='datasetHistTest', save=False, density=True, show=PREVIEW)

plot.barPlot2(np.arange(0,classesSize,1), yVal, xLabel='Classes', setXAxis= (-1,43),
             yLabel='Number of Images', fileName='datasetHistTest', save=True, show=PREVIEW)


# Plot Mean Value of Dataset Groups
xVal = ['Training', 'Validation', 'Testing']
yVal = [histTrain[0], histValid[0], histTest[0]]
plot.barPlot(xVal, yVal, xLabel='Dataset Groups',
             yLabel='Number of Images', fileName='datasetGroupsMean', save=True, show=PREVIEW)


# Plot Standard Deviation Value of Dataset Groups
xVal = ['Training', 'Validation', 'Testing']
yVal = [histTrain[1], histValid[1], histTest[1]]
plot.barPlot(xVal, yVal, xLabel='Dataset Groups',
             yLabel='Number of Images', fileName='datasetGroupsDeviation', save=True, show=PREVIEW)



# Preprocess Datasets

# Temporary Arrays
xTrainPreprocess = np.zeros(shape = (xTrain.shape[0] , xTrain.shape[1], xTrain.shape[2]))
xValidPreprocess = np.zeros(shape = (xValid.shape[0] , xValid.shape[1], xValid.shape[2]))
xTestPreprocess = np.zeros(shape = (xTest.shape[0] , xTest.shape[1], xTest.shape[2]))


# Preprocess Image Function Definition
def preprocessImage(image):
    
    grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.uint8)
    histogramEqualized= cv2.equalizeHist(grayscale)
    normalized = ((histogramEqualized - 128.0) / 128.0).astype(np.float32)

    return normalized


# Convert Datasets to float32
xTrain = xTrain.astype(np.float32)
xValid = xValid.astype(np.float32)
xTest  = xTest.astype(np.float32)

# Preprocess all Dataset Images
for i in tqdm(range(np.shape(xTrain)[0])):
    xTrainPreprocess[i] = preprocessImage(xTrain[i])
xTrain = xTrainPreprocess.reshape(xTrainPreprocess.shape + (1,))

for i in tqdm(range(np.shape(xValid)[0])):
    xValidPreprocess[i] = preprocessImage(xValid[i])
xValid = xValidPreprocess.reshape(xValidPreprocess.shape + (1,))

for i in tqdm(range(np.shape(xTest)[0])):
    xTestPreprocess[i] = preprocessImage(xTest[i])
xTest = xTestPreprocess.reshape(xTestPreprocess.shape + (1,))



# Tensors Setup

x = tf.placeholder(tf.float32,(None, np.shape(xTrain)[1], np.shape(xTrain)[2], np.shape(xTrain)[3])) # Inputs
y = tf.placeholder(tf.int32, (None)) # Labels 
oneHotY = tf.one_hot(y, classesSize) # One Hot Encoding

learningRate = tf.placeholder(tf.float32, (None)) # Learning Rate


if(TEST_NOW == True):
    logits = nn.neuralNetworkFull(x, classesSize) # Test Logits
else:
    logits = nn.neuralNetwork(x, classesSize) # Training Logits


# Define Cross Entropy Tensor
crossEntropy = tf.nn.softmax_cross_entropy_with_logits(labels=oneHotY, logits=logits)
# Define Loss Tensor
lossOperation = tf.reduce_mean(crossEntropy)
# Define Optimizer Tensor
optimizer = tf.train.AdamOptimizer(learning_rate = learningRate)
# Define Training Tensor
trainingOperation = optimizer.minimize(lossOperation)

# Prediction Check Tensor
correctPrediction = tf.equal(tf.argmax(logits, 1), tf.argmax(oneHotY, 1))
# Accuracy Tensor
accuracyOperation = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))


# Evaluate Training Step Result
def evaluate(xData, yData):
    examplesSize = len(xData)
    totalAccuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, examplesSize, BATCH_SIZE):
        batchX, batchY = xData[offset:offset+BATCH_SIZE], yData[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracyOperation, feed_dict={x: batchX, y: batchY})
        totalAccuracy += (accuracy * len(batchX))
    return totalAccuracy / examplesSize


# Evaluate Final Results
def evaluateFinal(xData, yData):
    examplesSize = len(xData)
    totalAccuracy = 0
    sess = tf.get_default_session()

    result = np.zeros(dtype=np.float64 ,shape=(classesSize,2))

    for offset in tqdm(range(0, examplesSize, 1)):
        batchX, batchY = xData[offset:offset+1], yData[offset:offset+1]
        accuracy = sess.run(accuracyOperation, feed_dict={x: batchX, y: batchY})
        label = batchY[0]

        # Count Right and Wrong Predictions Per Label
        if(accuracy == 1): 
            result[label,1] += 1
        else:
            result[label,0] += 1

    return result


# Define Saver Tensor
saver = tf.train.Saver()


# Run Test Dataset
if(TEST_NOW == True):
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.')) # Load Trained Model
        test_accuracy = evaluate(xTest, yTest)  # Evaluate Model
        print("Test Accuracy = {:.3f}".format(test_accuracy))
    exit()


# Max Accuracy While Training
maxAccuracy = 0.0

# Run Session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) # Initialize Global Variables
    examplesSize = len(xTrain)  # Dataset Size

    # History Variable for plot
    accuracyHistory = np.zeros([EPOCHS])
    
    # Learning Rate
    learning = LEARNING_RATE

    # Run Epochs
    for i in tqdm(range(EPOCHS)):
        xTrain, yTrain = shuffle(xTrain, yTrain)    # Shuffle dataset

        # Run Batch
        for offset in (range(0, examplesSize, BATCH_SIZE)):
            end = offset + BATCH_SIZE
            batchX, batchY = xTrain[offset:end], yTrain[offset:end]
            sess.run(trainingOperation, feed_dict={x: batchX, y: batchY, learningRate: learning})   #Train Model
        

        infoString = "EPOCH: {} -- ".format(i+1)

        validationAccuracy = evaluate(xValid,yValid)    # Evaluate Training Step
        infoString += "Accuracy: {:.3f}  -- ".format(validationAccuracy)

        # Save Model if it has a better accuracy
        if(validationAccuracy > maxAccuracy):
            maxAccuracy = validationAccuracy
            saver.save(sess, './lenet')
            infoString += "Saved: YES"
        else:
            infoString += "Saved: NO"

        print(infoString)

        # Save Accuracy value to history
        accuracyHistory[i] = validationAccuracy

    # Load Trained Model and Evaluate Final Model
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        accuracyResult = evaluateFinal(xValid, yValid)

    # Plot Accuracy Results Per Label
    yVal = (accuracyResult[:,1]*100)/(accuracyResult[:,0] + accuracyResult[:,1])
    plot.barPlot2(np.arange(0,classesSize,1), yVal, xLabel='Dataset Groups', setXAxis= (-1,43), setYAxis= (0,100),
             yLabel='Accuracy', fileName='AccuracyResults', save=True, show=PREVIEW)

    # Plot Accuracy Error Results Per Label
    yVal = (accuracyResult[:,0]*100)/(accuracyResult[:,0] + accuracyResult[:,1])
    plot.barPlot2(np.arange(0,classesSize,1), yVal, xLabel='Dataset Groups', setXAxis= (-1,43), setYAxis= (0,100),
             yLabel='Accuracy Error', fileName='AccuracyErrorResults', save=True, show=PREVIEW)

    # Plot Training Accuracy Per Training Step
    plot.linePlot(np.arange(1,EPOCHS+1,1), accuracyHistory, xLabel='EPOCH',yLabel='Accuracy', 
                            fileName='TrainingResult', save=True, show=PREVIEW)
    
    

    print("Max Accuracy: " + str(max(accuracyHistory)))
    



