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

[image7]: ./ "Preprocessing"


[image8]: ./ "Network Model"


[image1]: ./ "Datasets"
[image1]: ./ "Datasets"
[image1]: ./ "Datasets"
[image1]: ./ "Datasets"



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
| Layer             | Input             | Input Shape | Filter Shape | Filters | Stride | Output Shape | Activation Function |
|-------------------|-------------------|-------------|--------------|---------|--------|--------------|---------------------|
| Convolution 1     | Grayscale Image   | 32x32x1     | 5x5          | 32      | 1x1    | 28x28x32     | RELU                |
| MaxPool 1         | Convolution 1     | 28x28x32    |              |         | 2x2    | 14x14x32     |                     |
| Convolution 2     | MaxPool 1         | 14x14x32    | 5x5          | 32      | 1x1    | 10x10x64     | RELU                |
| MaxPool 2         | Convolution 2     | 10x10x64    |              |         | 2x2    | 5x5x64       |                     |
| Convolution 3     | MaxPool 2         | 5x5x64      | 5x5          | 32      | 1x1    | 1x1x128      | RELU                |


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


 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ?

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

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

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


