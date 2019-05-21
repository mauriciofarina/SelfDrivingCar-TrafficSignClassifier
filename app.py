import pickle
import numpy as np
import Plots as plot
import tensorflow as tf
from sklearn.utils import shuffle
import networks as nn
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt

PREVIEW = False  # Plot Preview On/Off


EPOCHS = 30
BATCH_SIZE = 128

rate = 0.001



print("**********************************\n")

# Dataset Files Path
training_file = '../data/train.p'
validation_file = '../data/valid.p'
testing_file = '../data/test.p'

# Open Dataset Files
with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

# Load Dataset Values
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print("Dataset Loaded\n")


print("**********************************\n")

# Number of training examples
n_train = np.shape(X_train)[0]

# Number of validation examples
n_validation = np.shape(X_valid)[0]

# Number of testing examples.
n_test = np.shape(X_test)[0]

# What's the shape of an traffic sign image?
image_shape = (np.shape(X_train)[1], np.shape(X_train)[2], np.shape(X_train)[3])

# List of Classes
classesList = np.unique(y_train)

# How many unique classes/labels there are in the dataset.
n_classes = np.shape(classesList)[0]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
print("\n")


print("**********************************\n")

'''

# Items Per Dataset Group
xVal = ['Training', 'Validation', 'Testing']
yVal = [n_train, n_validation, n_test]
plot.barPlot(xVal, yVal, xLabel='Dataset Groups',
             yLabel='Number of Images', fileName='datasetGroups', save=True, show=PREVIEW)


# Total of Items per Class in Training Dataset
xVal = y_train
histTrain = plot.histogramPlot(xVal, bins=n_classes, xLabel='Classes',
                   yLabel='Number of Images', fileName='datasetHistTrain', save=True, density=True, show=PREVIEW)

# Total of Items per Class in Validation Dataset
xVal = y_valid
histValid = plot.histogramPlot(xVal, bins=n_classes, xLabel='Classes',
                   yLabel='Number of Images', fileName='datasetHistValid', save=True, density=True, show=PREVIEW)

# Total of Items per Class in Testing Dataset
xVal = y_test
histTest = plot.histogramPlot(xVal, bins=n_classes, xLabel='Classes',
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

'''
print('Plots Done')

print("**********************************\n")

print('Start Preprocessing')

X_train = np.array(X_train, dtype = np.float32)
X_valid = np.array(X_valid, dtype = np.float32)
X_test = np.array(X_test, dtype = np.float32)

print('Normalizing Training Set')
for i in tqdm(range(np.shape(X_train)[0])):
    X_train[i] = (X_train[i] - 128.0) / 128.0

print('Normalizing Validation Set')
for i in tqdm(range(np.shape(X_valid)[0])):
    X_valid[i] = (X_valid[i] -128.0) / 128.0

print('Normalizing Testing Set')
for i in tqdm(range(np.shape(X_test)[0])):
    X_test[i] = (X_test[i] -128.0) / 128.0






print('Preprocessing Done')
print("**********************************\n")


x = tf.placeholder(tf.float32,(None, image_shape[0], image_shape[1], image_shape[2]))
y = tf.placeholder(tf.int32, (None))
one_hot_y = tf.one_hot(y, n_classes)

#logits = nn.LeNet(x, image_shape, n_classes, keep_prob = 0.5)
logits = nn.LeNetModified(x, image_shape, n_classes)

print('Tensors Defined')



cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)

print('Training Process Defined')

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

print('Validation Check Defined')


saver = tf.train.Saver()


print("**********************************\n")

print("Start Training\n")


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...\n")

    for i in tqdm(range(EPOCHS)):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in tqdm(range(0, num_examples, BATCH_SIZE)):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("\nEPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}\n".format(validation_accuracy))
        
    saver.save(sess, './lenet')
    print("Model saved")