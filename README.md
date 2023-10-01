# BezierSeg-Curves
This Python script performs various image processing tasks and includes a cancer detection module using a pre-trained model.
# Image Processing and Cancer Detection

This repository contains Python code for image processing, Bézier curve visualization, and cancer detection using a pre-trained deep learning model.

## Bézier Curve Visualization

The script `bezier_curve_visualization.py` uses the PIL library to load an image, visualizes Bézier curves on the image, and displays the result using Matplotlib.

### Usage
1. Ensure you have the required libraries installed: `PIL`, `matplotlib`, `numpy`.
2. Update the `image_path` variable with the path to your image.
3. Run the script.

## Image Cropping Tool

The file `image_cropping_tool.py` provides a simple GUI tool for cropping images. It utilizes the Tkinter library for the GUI and Matplotlib for image display.

### Usage
1. Run the script.
2. Click and drag to select an area for cropping.
3. The cropped image will be displayed on the right side, and a new file will be saved with a unique name.

## Cancer Detection with Pre-trained Model

The script `cancer_detection.py` uses TensorFlow and a pre-trained MobileNet V2 model from TensorFlow Hub to predict if an input image contains cancer.

### Usage
1. Install the required libraries: `tensorflow`, `tensorflow_hub`.
2. Update the `image_path` variable with the path to your image.
3. Run the script.

Note: The MobileNet V2 model used for cancer detection is a pre-trained model on ImageNet. Make sure to replace it with a more suitable model for cancer detection.

Feel free to explore and modify the code based on your requirements!
