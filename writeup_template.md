## Writeup
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_notcar.png
[image2]: ./examples/car_notcar_hog.jpg
[image3]: ./examples/color_spaces.png
[image4]: ./examples/sw1.png
[image5]: ./examples/sw2.png
[image6]: ./examples/multiples.png
[image7]: ./examples/heatpoint.png
[image8]: ./examples/final.png
[video1]: ./project.mp4
[video2]: ./test.mp4
[video3]: ./test_own.mp4


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Project Segmentation

My project consists in the following main points :

* 1. Data Exploration
* 2. Features extraction
* 3. Model Training
* 4. Sliding Window Search & Heat Map

### 1. Data Exploration

The goal of the project is to detect vehicles on a video. To do so, we first need vehicle and non vehicle images. 
Here are two links to where I found the images : 
https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip
https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip

When we explore the data a little bit, we can see that there are 8792  cars and 8968 non-cars PNG (64x64x3) images.
We can randomly plot one from each to see what our pictures are like.

![alt text][image1]


### 2. Features extraction

#### A. Main functions

When we know what our data looks like, we can start the extraction. The goal of the extraction is to identify what bests describes a car and put it into a classifier.
I wrote 4 functions to extract 3 kinds of features :
* ` bin_spatial(img, size=(32, 32)) ` is in charge to resize the images to (32,32) (that can be changed) not to have too heavy files and still keep the features. When we resize an image from (64,64) to (32,32) we want to make sure that the properties that make a car recognizable from a road or a tree are still there. We then use the `ravel()` function to convert the images into features.

* `color_hist(img, nbins=32, bins_range=(0, 256))` is computing color histogram depending on the image and the color space. What is sent in entry to this function is the image converted to the desired color space. The output are the features of color  histogram.

* `get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True)` is a function that returns hog features. Here is an example of hog features for our two images :
![alt text][image2]

* Finally, I wrote a function `extract_features(imgs, color_space='RGB', spatial_size=(32, 32),hist_bins=32, orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True)` that uses a lot of parameters and calls the three previous functions. Once it gets the three features independently, it stacks them into one feature vector that we will send to our classifier..

####  B. Parameters

It wasn't that easy to find the rights parameters to define features. If the parameters are not well defined, we can have the wrong idea of features. Here are my parameters :
`
color_space = 'YCrCb' # Color space
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [350, 656] # Min and max in y to search in slide_window()
`

The first parameter is the color space. I decided to extract the features on yCrCb because I had good results with it, better than HSV. 
![alt text][image3]

Orientations, pix_per_cell and cell_per_block are parameters that have been tweaked.
Concerning the hog channels, I also tried all four parameters but training on all channels seemed better than training on only one channel. When we convert the image and compute the histogram, we realize that all channels can be needed.
We then have spatial_size that was defined to (16,16). It is roughly assimilated to the size of the windows we are going to have so it needs to be large enough to fit a car in it but not too large to be precise on what was detected.
Hist_bins helps measure my vector length.

Finally, we decide if we want to have feature extraction or not depending on the type of the features (spatial, histogram, hog).

### 3. Model Training

For training a model and teaching it to learn how to recognize a car image from a non car image, we needed the features extraction first. Features are what defined a car and a non car. We process the features extraction to both cars and non-cars images because we also want the model to learn what is not a car. We can go a little into the code to analize how training works :
#### A. Features / Labels
We first have two vectors in which we concatenate the features together. We associate that to every labels we already have, by labels I mean "Car" or "NotCar".
`
-- Define the features vector
X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
-- Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
`
#### B. Normalization
We normalize the X input to have better range and generalisation.
`
-- Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
-- Apply the scaler to X
scaled_X = X_scaler.transform(X)
`
#### C. Random Split
We shuffle our data and split the data into Training/Test Sets. That will be used to verify our model later.
`
-- Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)
`

#### D . Classifier definition
I decided to use a linear SVC because it was advised by Udacity's course but also because it was really easy to develop, just two lines of code, and was really appropriate for the project. I chose a C value of 2.0 to have a little bit of learning on the exceptions but not too much, and a linear Kernel.
`
-- Use a linear SVC 
svc = SVC(C=2.0, kernel = 'linear')
svc.fit(X_train, y_train)
`
After training, my model returns an accuracy of 98.7% on the test set.

### 4. Sliding Window Search & Heat Map

I finally implemented a sliding window function that will scroll the image, identify the features of a car or not, and decided if it is a car or not.
I have used four functions :
* `convert_color(img, conv ='YCrCb')`, a simple color converted.
* `draw_labeled_bboxes(img, labels)` that draws rectangles on the image depending on where the labels are.
* `apply_threshold(heatmap, threshold)` that used the heatmap technique to remove false-positives.
* `find_cars(img, color_space, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)` is the function that implements sliding window and heat map.

#### A. Find_Cars function breakdown
This will not be a line by line breakdown but a general understanding of what the function does and returns.

* First, we define a region of interest in which to look for the cars. The region was previously defined when I was explaining the parameters tweaking. y_start_stop parameter is the one that helps us find cars only on the [350,650] area. This is only to be faster on implementation. 
* Then, we convert our color to the color space we want. I chose YCrCb.
* We then define how to look for the features. Initially, I implemented a function that would extract hog features on every box. The find cars function allows us to extract the features once for a faster and more permormant implementation. I define steps and boxes according to the parameters I sent to the function (orient, pix_per_cell, cell_per_block). Before going into the blocks, I extract the hog features.
* Once we have that, we implement two for loops that will look into every block for the features we want and finally use the classifier with `test_prediction = svc.predict(test_features)` (test_features) is my computation of the 3 known features. 
* When the prediction is equal to 1, we draw a rectangle and add a heat point to the heat map.

With this, we now have multiple rectangles depending on the sliding window blocks used.
![alt text][image6]


#### B. Pipeline Implementation

In order to remove false positives on the left and to have one rectangle for the car only, we use our apply_threshold function. We send to that function the heatmap and a threshold we want to apply. I used a threshold of 1.1 by experimentation.
![alt text][image7]

We then have a heatmap where the false positive are removed by thresholding.

We finally call the draw_labeled_bboxes function to draw rectangles only where the heat points remains.

![alt text][image8]

#### C. Outputs
Here are my outputs.

![alt text][image4]
![alt text][image5]
---

### Video Implementation

Here's a [link to my video result](./project.mp4)
Here's a [link to my test result](./test.mp4)
Here's a [link to my own video result](./test_own.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

* I tried the pipeline function on my own video, it didn't work. The reason for that might be the size of the squares of my sliding window. Maybe I was too close. In fact, what we need would be for the sliding window to adapt to whether you are far or close from the cars. If you have another reason, please provide some help.
* I would like to add lane lines detection on it.
* We still have false-positives.
* The pipeline would work better using deep learning techniques I suppose. 
