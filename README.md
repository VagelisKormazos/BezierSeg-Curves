# BezierSeg-Curves
This Python script performs various image processing tasks and includes a cancer detection module using a pre-trained model.
# Image Processing and Cancer Detection

This repository contains Python code for image processing, Bézier curve visualization, and cancer detection using a pre-trained deep learning model.

![image](https://github.com/VagelisKormazos/BezierSeg-Curves/assets/100516014/1fa5a193-f843-4dbe-a8ad-4fd5ebc86fe9)


## Bézier Curve Visualization

The script `bezier_curve_visualization.py` uses the PIL library to load an image, visualizes Bézier curves on the image, and displays the result using Matplotlib.

### Usage
1. Ensure you have the required libraries installed: `PIL`, `matplotlib`, `numpy`.
2. Update the `image_path` variable with the path to your image.
3. Run the script.

## Image Cropping Tool

The file `image_cropping_tool.py` provides a simple GUI tool for cropping images. It utilizes the Tkinter library for the GUI and Matplotlib for image display.

![image](https://github.com/VagelisKormazos/BezierSeg-Curves/assets/100516014/23709e4b-948e-4879-bbce-e8827347e77a)


## Cancer Detection with Pre-trained Model

The script `cancer_detection.py` uses TensorFlow and a pre-trained MobileNet V2 model from TensorFlow Hub to predict if an input image contains cancer.
![image](https://github.com/VagelisKormazos/BezierSeg-Curves/assets/100516014/6509993c-eba7-4f4a-8505-ffeb5ba5913f)
The model recognize the cancer:
![image](https://github.com/VagelisKormazos/BezierSeg-Curves/assets/100516014/974c193a-87da-4ccd-ba3d-6abdd9e5b664)

# Image Processing and Cancer Detection

This Python script performs various image processing tasks and includes a cancer detection module using a pre-trained model.

## What You'll Learn

This project provides hands-on experience in image processing, graphical user interface (GUI) development, and deep learning-based image classification. By working with this code, you'll gain proficiency in the following areas:

### 1. Image Processing

- **Bézier Curve Visualization:** Understand how to implement and visualize Bézier curves on images using control points.
  
- **Image Cropping Tool:** Learn to use tkinter for creating a GUI tool to interactively select and save cropped regions from an image.

### 2. GUI Development

- **Tkinter Integration:** Gain experience in integrating the Tkinter library to create a user-friendly interface for image processing tasks.

### 3. Deep Learning

- **TensorFlow and TensorFlow Hub:** Utilize TensorFlow and TensorFlow Hub to load a pre-trained deep learning model (MobileNet V2) for image classification.

- **Model Inference:** Learn how to preprocess images and make predictions using a pre-trained model for tasks such as cancer detection.

### 4. Project Organization

- **GitHub Repository:** Understand how to structure and document a project on GitHub, including README files, code organization, and version control.

### 5. Problem Solving

- **Debugging:** Develop debugging skills by troubleshooting potential issues that may arise during script execution.

### 6. Collaboration and Contribution

- **Contributing to Open Source:** Explore how to contribute to open-source projects by submitting improvements, bug fixes, or new features.

By engaging with this project, you'll enhance your skills in image manipulation, GUI development, and deep learning, preparing you for a variety of tasks in computer vision and software development.

## Getting Started

### Prerequisites

Make sure you have the following libraries installed:

- PIL
- matplotlib
- numpy
- ssl
- tkinter
- tensorflow
- tensorflow_hub

Install missing libraries using:

```bash
pip install pillow matplotlib numpy tensorflow tensorflow_hub


### Usage
1. Install the required libraries: `tensorflow`, `tensorflow_hub`.
2. Update the `image_path` variable with the path to your image.
3. Run the script.

Note: The MobileNet V2 model used for cancer detection is a pre-trained model on ImageNet. Make sure to replace it with a more suitable model for cancer detection.

Feel free to explore and modify the code based on your requirements!
