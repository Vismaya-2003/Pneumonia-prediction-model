import tensorflow as tf
from tensorflow.keras.models import load_model
import streamlit as st
import numpy as np

# Load the trained model
model_path = r'C:\Users\visma\OneDrive\Desktop\pneumonia\Pneumonia-prediction.keras'
model = load_model(model_path)

# Define categories
data_cat = ['NORMAL', 'PNEUMONIA']

# Image dimensions
img_height = 180
img_width = 180

# Load and preprocess the image
image_path = st.text_input("Please input the image to check for pneumonia:",'image1.jpeg')
st.header("Pneumonia prediction model")
image_load = tf.keras.utils.load_img(image_path, target_size=(img_height, img_width))
img_arr = tf.keras.utils.img_to_array(image_load)  # Convert the image to an array
img_bat = tf.expand_dims(img_arr, 0)  # Add batch dimension

# Make a prediction
predict = model.predict(img_bat)

# Apply softmax to get probabilities
score = tf.nn.softmax(predict[0])

# Display the image using Streamlit
st.image(image_path, width=200)


# Display the result with the predicted class and accuracy
st.write("The scan shows a {} chest with an accuracy of {:.2f}%".format(data_cat[np.argmax(score)], np.max(score) * 100))
