# Pneumonia Prediction Model with CNN and Streamlit

## Overview

This project implements a Convolutional Neural Network (CNN) model for predicting pneumonia from chest X-ray images. The trained model is deployed via a Streamlit web application, providing an easy-to-use interface for real-time predictions.

## Features

- **Convolutional Neural Network (CNN)**: Classifies chest X-ray images into 'pneumonia' or 'non-pneumonia'.
- **Streamlit Web Application**: Allows users to upload images and receive immediate predictions.
- **Real-time Prediction**: Instant results based on the CNN model.

## Technologies

- **Python**: Programming language used for model training and web application development.
- **TensorFlow/Keras**: Libraries for building and training the CNN model.
- **Streamlit**: Framework for creating the web application interface.

## Dataset
- The dataset was acquired from Kaggle [Chest X-Ray Pneumonia Dataset on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

## Usage

1. **Run the Streamlit app**:
   streamlit run app.py
2. **Open the provided local URL in your web browser**
3. **Upload a chest X-ray image using the web interface**
4. **View the prediction results directly on the webpage**

## Project Structure
- **app.py**: Streamlit application code.
- **pneumonia.ipynb**: Contains the CNN model architecture and training code.
