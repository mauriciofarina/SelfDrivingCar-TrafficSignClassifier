# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The steps of this project are the following:
1. Load the data set
1. Explore, summarize and visualize the data set
1. Preprocess Datasets
1. Design, train and test a model architecture
1. Use the model to make predictions on new images
1. Analyze the softmax probabilities of the new images


[//]: # (Image References)

[image1]: ./plots/datasetGroups.png "Datasets"
[image2]: ./plots/datasetHistTrain.png "Training Dataset"
[image3]: ./plots/datasetHistValid.png "Validation Dataset"
[image4]: ./plots/datasetHistTest.png "Testing Dataset"
[image5]: ./plots/datasetGroupsMean.png "Datasets Mean"
[image6]: ./plots/datasetGroupsDeviation.png "Datasets Standard Deviation"
[image7]: ./plots/preprocess.png "Preprocessing"
[image8]: ./plots/model.png "Network Model"
[image9]: ./plots/AccuracyResultsTrain.png "Accuracy Train"
[image10]: ./plots/AccuracyErrorResultsTrain.png "Error Train"
[image11]: ./plots/AccuracyResultsValid.png "Accuracy Valid"
[image12]: ./plots/AccuracyErrorResultsValid.png "Error Valid"
[image13]: ./plots/AccuracyResultsTest.png "Accuracy Test"
[image14]: ./plots/AccuracyErrorResultsTest.png "Error Test"
[image15]: ./plots/TrainingResult.png "Training Results"




---
## Development

### Development Files

| File | Description |
| ------ | ------ |
| Plots.py | Predefined Plot Functions |
| networks.py | Neural Network Model Implementations | 
| trainer.json | Train and Evaluate Models | 
| imageClassifier.py | Classify Images |

### Data Set Summary & Exploration


After loading the provided dataset files, the following results were found:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

![alt text][image1]


The sample distribution for each dataset and its statistics are presented below:

#### Training Dataset
![alt text][image2]

#### Validation Dataset
![alt text][image3]

#### Testing Dataset
![alt text][image4]

#### Datasets Means
![alt text][image5]

#### Datasets Standard Deviations
![alt text][image6]





### Proprocess

In order to obtain better results, 3 steps of preprocessing was implemented:

#### 1. Grayscale Conversion

After multiple tests with differente color maps and channels, it was notice that a single channel grayscale image obtained the best results for this model.

#### 2. Histogram Equalization

In order to ajust light conditions, the histogram equalizatiton was applyied to the grayscale images

#### 3. Normalization

Since data should be normalized for better results, each image was normalized by the equation `(Pixel_Value -128)/128`

#### Preprocessing Results

In the image below, a original image and its 3 preprocess steps can be observed:

![alt text][image7]




### Model Architecture

The designed model architecture can be seen on the image below:

![alt text][image8]


The final model consisted of the following layers:


#### Convolution
| Layer         | Input           | Input Shape | Filter | Stride | Output Shape | Activation Function |
|---------------|-----------------|-------------|--------|--------|--------------|---------------------|
| Convolution 1 | Grayscale Image | 32x32x1     | 5x5x32 | 1x1    | 28x28x32     | RELU                |
| MaxPool 1     | Convolution 1   | 28x28x32    |        | 2x2    | 14x14x32     |                     |
| Convolution 2 | MaxPool 1       | 14x14x32    | 5x5x32 | 1x1    | 10x10x64     | RELU                |
| MaxPool 2     | Convolution 2   | 10x10x64    |        | 2x2    | 5x5x64       |                     |
| Convolution 3 | MaxPool 2       | 5x5x64      | 5x5x32 | 1x1    | 1x1x128      | RELU                |


#### Fully Connected
| Layer             | Input             | Inputs | Outputs | Activation Function |
|-------------------|-------------------|--------|---------|---------------------|
| Fully Connected 1 | Convolution 3     | 128    | 1024    | RELU                |
| Fully Connected 2 | Fully Connected 1 | 1024   | 256     | RELU                |
| Fully Connected 3 | Fully Connected 2 | 256    | 43      |                     |


#### Dropout

In order to prevent overfitting, a dropout regularization added to the model:

| Layer             | Keep Probability |
|-------------------|------------------|
| Convolution 1     | 0.9              |
| Convolution 2     | 0.9              |
| Convolution 3     | 0.9              |
| Fully Connected 1 | 0.8              |
| Fully Connected 2 | 0.8              |


 ### Model Traning
 
 In order to obtain the final trained model, multiple tests were executed in order to find the best results. The provided learning rate of 0.001 and batch size of 128 resulted in good results and were not changed. In the other hand, the number of epochs was too short for obtaining the necessary 0.93 accuracy, so it was changed to 50 epochs.
 
 | Hyperparameter   | Value |
 |------------------|-------|
 | Learning Rate    | 0.001 |
 | Batch Size       | 128   |
 | Number of Epochs | 50    |
 
 
 After some tests, it was noticed that many times, the best accuracy was not found on the last epoch. In order to fix that, the training process was changed in a way that it always saves the best accuracy found, instead of the last epoch. 
 
 The training process was then executed, resulting in the following results:
 
 #### Max Accuracy Epoch: 32
 
  ![alt text][image15]
 
 
 #### Training Set Accuracy: 0.9991
 
 ![alt text][image9]
 
 ![alt text][image10]
 
 #### Validation Set Accuracy: 0.9678
 
 ![alt text][image11]
 
 ![alt text][image12]
 
 #### Test Set Accuracy: 0.9393
 
 ![alt text][image13]
 
 ![alt text][image14]
 
 
 
 

 
 
 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:


The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 




