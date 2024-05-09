import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model

# Step 1: Load Image Data
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

# Step 4: Choose a Model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(5, activation='softmax')  # 5 output classes (assuming 5 flower species)
])

# Step 5: Compile the Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 6: Train the Model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=10)

# Step 7: Evaluate the Model
test_loss, test_accuracy = model.evaluate(validation_generator, steps=validation_generator.samples // batch_size)
print("Test Accuracy:", test_accuracy)

# Step 8: Save the Model
model.save("flower_classifier_model.h5")

# Step 9: Load the Model (for making predictions)
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
image_path = './360_F_105573812_cvD4P5jo6tMPhZULX324qUYFbNpXlisD.jpg'  # Provide the path to your image
predicted_flower = predict_flower(image_path)
print("Predicted Flower:", predicted_flower)
