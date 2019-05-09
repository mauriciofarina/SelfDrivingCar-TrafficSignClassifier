import pickle
import numpy as np
import matplotlib.pyplot as plt

print("**********************************\n")

# Dataset Files Path
training_file = '../data/train.p'
validation_file= '../data/valid.p'
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
image_shape = (np.shape(X_train)[1], np.shape(X_train)[2])

# How many unique classes/labels there are in the dataset.
n_classes = np.shape(y_train)[0]

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
print("\n")


print("**********************************\n")



plt.plot(['Training','Validation', 'Testing'], [n_train, n_validation, n_test])
plt.xlabel('Datasets')
plt.ylabel('Images')
plt.savefig('plots/.png')