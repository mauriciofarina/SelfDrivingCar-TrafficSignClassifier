import numpy as np
import time

np.warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
os.environ["QT_QPA_PLATFORM"] = "offscreen"
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


EPOCHS = 20
BATCH_SIZE = 128
LEARNING_RATE = 0.001
PLOT_HIGHT=30
TEST_NOW = False

for index, value in enumerate(sys.argv):
    
    if value == "-e":
        EPOCHS = int(sys.argv[index+1])
    
    elif value == "-b":
        BATCH_SIZE = int(sys.argv[index+1])

    elif value == "-l":
        LEARNING_RATE = float(sys.argv[index+1])
    elif value == "-p":
        PLOT_HIGHT = int(sys.argv[index+1])
    elif value == "-t":
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

print("Dataset Loaded")



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

'''
print("Number of training examples =", trainSize)
print("Number of testing examples =", testSize)
print("Image data shape =", imageShape)
print("Number of classes =", classesSize)
print("\n")
'''



# Items Per Dataset Group
xVal = ['Training', 'Validation', 'Testing']
yVal = [trainSize, validationSize, testSize]
plot.barPlot(xVal, yVal, xLabel='Dataset Groups',
             yLabel='Number of Images', fileName='datasetGroups', save=True, show=PREVIEW)


# Total of Items per Class in Training Dataset
xVal = yTrain
histTrain = plot.histogramPlot(xVal, bins=classesSize, xLabel='Classes',
                   yLabel='Number of Images', fileName='datasetHistTrain', save=True, density=True, show=PREVIEW)

# Total of Items per Class in Validation Dataset
xVal =yValid
histValid = plot.histogramPlot(xVal, bins=classesSize, xLabel='Classes',
                   yLabel='Number of Images', fileName='datasetHistValid', save=True, density=True, show=PREVIEW)

# Total of Items per Class in Testing Dataset
xVal = yTest
histTest = plot.histogramPlot(xVal, bins=classesSize, xLabel='Classes',
                   yLabel='Number of Images', fileName='datasetHistTest', save=True, density=True, show=PREVIEW)


# Mean Value of Dataset Groups
xVal = ['Training', 'Validation', 'Testing']
yVal = [histTrain[0], histValid[0], histTest[0]]
plot.barPlot(xVal, yVal, xLabel='Dataset Groups',
             yLabel='Number of Images', fileName='datasetGroupsMean', save=True, show=PREVIEW)


# Standard Deviation Value of Dataset Groups
xVal = ['Training', 'Validation', 'Testing']
yVal = [histTrain[1], histValid[1], histTest[1]]
plot.barPlot(xVal, yVal, xLabel='Dataset Groups',
             yLabel='Number of Images', fileName='datasetGroupsDeviation', save=True, show=PREVIEW)


print('Plots Done')


print('Start Preprocessing...')


print('Converting to Grayscale')

xTrainPreprocess = np.zeros(dtype = np.uint8, shape = (xTrain.shape[0] , xTrain.shape[1], xTrain.shape[2]))
xValidPreprocess = np.zeros(dtype = np.uint8, shape = (xValid.shape[0] , xValid.shape[1], xValid.shape[2]))
xTestPreprocess = np.zeros(dtype = np.uint8, shape = (xTest.shape[0] , xTest.shape[1], xTest.shape[2]))


for i in tqdm(range(np.shape(xTrain)[0])):
    xTrainPreprocess[i] = cv2.cvtColor(xTrain[i], cv2.COLOR_RGB2GRAY)
    xTrainPreprocess[i] = cv2.equalizeHist(xTrainPreprocess[i])
    xTrainPreprocess[i] = (xTrainPreprocess[i] - 128.0) / 128.0

for i in tqdm(range(np.shape(xValid)[0])):
    xValidPreprocess[i] = cv2.cvtColor(xValid[i], cv2.COLOR_RGB2GRAY)
    xValidPreprocess[i] = cv2.equalizeHist(xValidPreprocess[i])
    xValidPreprocess[i] = (xValidPreprocess[i] - 128.0) / 128.0

for i in tqdm(range(np.shape(xTest)[0])):
    xTestPreprocess[i] = cv2.cvtColor(xTest[i], cv2.COLOR_RGB2GRAY)
    xTestPreprocess[i] = cv2.equalizeHist(xTestPreprocess[i])
    xTestPreprocess[i] = (xTestPreprocess[i] - 128.0) / 128.0




print('Defining Input Dataset')

channels = 1

xTrain = np.zeros(dtype = np.float32, shape = (xTrain.shape[0] , xTrain.shape[1], xTrain.shape[2], channels))
xValid = np.zeros(dtype = np.float32, shape = (xValid.shape[0] , xValid.shape[1], xValid.shape[2], channels))
xTest = np.zeros(dtype = np.float32, shape = (xTest.shape[0] , xTest.shape[1], xTest.shape[2], channels))



xTrain[:,:,:,0] = xTrainPreprocess
xValid[:,:,:,0] = xValidPreprocess
xTest[:,:,:,0] = xTestPreprocess

imageShape = (np.shape(xTrain)[1], np.shape(xTrain)[2], np.shape(xTrain)[3])





print('Preprocessing Done')


print('TensorFlow Setup...')



x = tf.placeholder(tf.float32,(None, imageShape[0], imageShape[1], imageShape[2]))
y = tf.placeholder(tf.int32, (None))
oneHotY = tf.one_hot(y, classesSize)



logits = nn.LeNetModified2(x, classesSize)


print('Tensors Defined')



crossEntropy = tf.nn.softmax_cross_entropy_with_logits(labels=oneHotY, logits=logits)
lossOperation = tf.reduce_mean(crossEntropy)

learningRate = tf.placeholder(tf.float32, (None))

optimizer = tf.train.AdamOptimizer(learning_rate = learningRate)
trainingOperation = optimizer.minimize(lossOperation)

print('Training Process Defined')

correctPrediction = tf.equal(tf.argmax(logits, 1), tf.argmax(oneHotY, 1))
accuracyOperation = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))



def evaluate(xData, yData):
    examplesSize = len(xData)
    totalAccuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, examplesSize, BATCH_SIZE):
        batchX, batchY = xData[offset:offset+BATCH_SIZE], yData[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracyOperation, feed_dict={x: batchX, y: batchY})
        totalAccuracy += (accuracy * len(batchX))
    return totalAccuracy / examplesSize


def evaluate2(xData, yData):

    print("Evaluating Results..")
    result = np.zeros(shape=(classesSize,2))
    examplesSize = len(xData)
    totalAccuracy = 0
    sess = tf.get_default_session()
    for offset in tqdm(range(0, examplesSize, 1)):
        batchX, batchY = xData[offset:offset+1], yData[offset:offset+1]
        accuracy = sess.run(accuracyOperation, feed_dict={x: batchX, y: batchY})

        label = batchY[0]
        if(accuracy == 1):
            result[label,1] += 1
        else:
            result[label,0] += 1

    return result

print('Validation Check Defined')


saver = tf.train.Saver()



print("\nStart Training\n")


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    examplesSize = len(xTrain)
    
    print("Training...\n")

    accuracyHistory = np.zeros([EPOCHS])

    learning = LEARNING_RATE

    for i in (range(EPOCHS)):

        startTime = time.time()
        xTrain, yTrain = shuffle(xTrain, yTrain)
        for offset in tqdm(range(0, examplesSize, BATCH_SIZE)):
            end = offset + BATCH_SIZE
            batchX, batchY = xTrain[offset:end], yTrain[offset:end]
            sess.run(trainingOperation, feed_dict={x: batchX, y: batchY, learningRate: learning})
            
        validationAccuracy = evaluate(xValid,yValid)

    

        endTime = time.time()
        deltaTime = endTime - startTime

        accuracyHistory[i] = validationAccuracy

        infoString = "EPOCH: {} -- ".format(i+1)
        infoString += "Validation Accuracy: {:.3f}  -- ".format(validationAccuracy)
        infoString += "Runtime: {:.3f}s -- ".format(deltaTime)
        infoString += "Learning Rate: {:.7f}".format(learning)
        print(infoString)

        # if(((i+1)%5) == 0):
        #     learning = learning*(3/4)


        if(i == EPOCHS-1):
            accuracyResult = evaluate2(xValid,yValid)


    yVal = (accuracyResult[:,1]*100)/(accuracyResult[:,0] + accuracyResult[:,1])
    plot.barPlot2(np.arange(1,classesSize,1), yVal, xLabel='Dataset Groups', setXAxis= (-1,43),
             yLabel='Accuracy', fileName='AccuracyResults', save=True, show=PREVIEW)

    yVal = (accuracyResult[:,0]*100)/(accuracyResult[:,0] + accuracyResult[:,1])
    plot.barPlot2(np.arange(1,classesSize,1), yVal, xLabel='Dataset Groups', setXAxis= (-1,43), setYAxis= (0,100),
             yLabel='Accuracy Error', fileName='AccuracyErrorResults', save=True, show=PREVIEW)


    plot.linePlot(np.arange(1,EPOCHS+1,1), accuracyHistory, xLabel='EPOCH',yLabel='Accuracy', fileName='TrainingResult', save=True, show=PREVIEW)
        
    #saver.save(sess, './lenet')
    print("Model saved")

    print("Max Accuracy: " + str(max(accuracyHistory)))
    

if(TEST_NOW == True):
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))

        test_accuracy = evaluate(xTest, yTest)
        print("Test Accuracy = {:.3f}".format(test_accuracy))
