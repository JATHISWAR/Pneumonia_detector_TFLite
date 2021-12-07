# Android Covid Pneumonia Detector using TFlite 

## Overview

This is an example application for [TensorFlow Lite](https://tensorflow.org/lite)
on Android. It uses
[Image classification](https://www.tensorflow.org/lite/models/image_classification/overview)
to continuously classify whatever it sees from the device's back camera.
Inference is performed using the TensorFlow Lite Java API. The demo app
classifies frames in real-time, displaying the top most probable
classifications. It allows the user to choose between a floating point or
[quantized](https://www.tensorflow.org/lite/performance/post_training_quantization)
model, select the thread count, and decide whether to run on CPU,GPU or via a neural network API


## Requirements

*   Android Studio 3.2 

*   Android device in
    [developer mode](https://developer.android.com/studio/debug/dev-options)
    with USB debugging enabled

*   USB cable (to connect Android device to your computer)


### Model 

The model trained is a simple CNN model done using python in Tensorflow Framework.
You can download the dataset at https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
To convert from TF to TFlite you can use https://www.youtube.com/watch?v=MZx1fhbL2q4


<img src="https://github.com/JATHISWAR/Pneumonia_detector_TFLite/blob/master/Screenshot%202020-09-06%20at%201.01.20%20PM.png" width="200" height="400">

<img src="https://github.com/JATHISWAR/Pneumonia_detector_TFLite/blob/master/Screenshot%202020-09-06%20at%201.01.38%20PM.png" width="200" height="400">

<img src="https://github.com/JATHISWAR/Pneumonia_detector_TFLite/blob/master/Screenshot%202020-09-06%20at%201.01.48%20PM.png" width="200" height="400">



