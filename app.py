import pickle
import numpy as np

training_file = '../data/train.p'
validation_file= '../data/valid.p'
testing_file = '../data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print("Dataset Loaded")



# TODO: Number of training examples
n_train = np.shape(X_train)

# TODO: Number of validation examples
n_validation = np.shape(X_valid)

# TODO: Number of testing examples.
n_test = np.shape(X_test)

# TODO: What's the shape of an traffic sign image?
image_shape = X_test

# TODO: How many unique classes/labels there are in the dataset.
n_classes = X_test

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)