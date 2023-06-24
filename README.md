# potato_disease_detection

<img width="442" alt="Screenshot 2023-06-24 at 08 36 54" src="https://github.com/sumit-jr/potato_disease_detection/assets/81641001/79f162b0-02bb-4ba6-8779-5cf9001fc455">

This repository contains the implementation of a Convolutional Neural Network (CNN) for potato disease detection. The CNN model is trained on a labeled dataset of potato plant images to accurately classify various diseases, helping farmers identify and mitigate potential crop losses.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)


## Introduction

Potato diseases can have a significant impact on crop yield and quality. Timely identification and appropriate treatment of diseases are crucial for effective disease management. This project aims to develop a machine learning solution using CNNs to automate the detection of potato diseases.

By leveraging the power of deep learning and image analysis techniques, this project provides a tool that can accurately classify potato diseases based on input images. The model can be used by farmers, agronomists, or researchers to assess the health status of potato plants and take necessary actions to prevent disease spread and minimize crop damage.

## Installation

To use the potato disease detection model, follow the installation steps below:

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/potato-disease-detection.git
   cd potato-disease-detection

   Set up a virtual environment (optional but recommended):
   ```python3 -m venv env
      source env/bin/activate  # (for Linux/macOS)
      env\Scripts\activate  # (for Windows)


## Usage
To use the potato disease detection model, follow these steps:

2. Prepare the input image:

Ensure that the image you want to classify is in a supported format (e.g., JPEG, PNG).
If necessary, resize or preprocess the image to match the expected input size of the model.
Run the prediction script:

```python predict.py --image <path_to_image>```

Replace <path_to_image> with the path to the image you want to classify.

3. Interpret the results:

The script will output the predicted disease class along with the confidence score.
If the confidence score is below a certain threshold, the model may indicate that the image does not contain any significant disease symptoms.

## Dataset
The training and evaluation of the potato disease detection model rely on a labeled dataset of potato plant images. The dataset used in this project contains images of healthy potato plants as well as those affected by various diseases.

The link for the dataset is :- [Dataset](https://github.com/sumit-jr/potato_disease_detection/tree/master/PlantVillage)

[Zip Dataset](https://github.com/sumit-jr/potato_disease_detection/tree/master/PlantVillage)

## Model Architecture
The potato disease detection model is based on a Convolutional Neural Network (CNN) architecture. CNNs are well-suited for image classification tasks, as they can automatically learn meaningful features from images.

The architecture used in this project consists of multiple convolutional layers, followed by pooling layers for dimensionality reduction. The extracted features are then flattened and passed through fully connected layers to perform classification. The specific architecture and hyperparameters can be found in the model implementation files.

## Training
To train the potato disease detection model from scratch, follow these steps:

1. Prepare the dataset:

Obtain a labeled dataset of potato plant images with disease annotations.
Split the dataset into training and validation sets.

2. Configure the training parameters:

Adjust the hyperparameters in the training script, such as learning rate, batch size, and number of epochs.
Optionally, modify the data augmentation techniques or other training strategies.

3. Start the training process:

4. Monitor the training progress:

The script will display the loss and accuracy metrics during training.
Additionally, you can visualize the training curves using tensorboard or other plotting tools.

5. Save the trained model:

Once the training is complete, save the trained model weights for later use.


## Evaluation
The performance of the potato disease detection model can be evaluated on a separate test dataset. To evaluate the model, follow these steps:

1. Prepare the test dataset:

Collect a set of potato plant images that were not used during training or validation.
Ensure that the test dataset includes a representative distribution of healthy plants and various disease classes.

2. Run the evaluation

3. Analyze the evaluation results:

The script will output various evaluation metrics, such as accuracy, precision, recall, and F1-score, for each disease class and overall performance.

## Contributing
Contributions to this project are welcome! If you encounter any issues, have ideas for improvements, or want to contribute new features, please feel free to submit a pull request or open an issue in the repository.


