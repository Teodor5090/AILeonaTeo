import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model

data_dir = './flowers'
batch_size = 32
image_size = (150, 150)
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='training')

validation_generator = train_datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation')


model = load_model("flower_classifier_model.h5")

# Step 10: Function to predict flower type from an image file
def predict_flower(image_path):
    img = load_img(image_path, target_size=image_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    class_labels = train_generator.class_indices
    class_labels = dict((v, k) for k, v in class_labels.items())
    predicted_class_label = class_labels[predicted_class]
    return predicted_class_label

# Step 11: Provide an image path and get the prediction
image_path = 'Sunflower_sky_backdrop.jpg'  # Provide the path to your image
predicted_flower = predict_flower(image_path)
print("Predicted Flower:", predicted_flower)