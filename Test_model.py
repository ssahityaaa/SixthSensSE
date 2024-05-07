import tensorflow as tf
import numpy as np
from PIL import Image
import os
import PyPDF2
import pytesseract

class_names = ["ADITI", "KRATIKA", "Radhika", "SHREYA", "Sahitya"]

model_path = "handwriting_recognition_model2.h5"
loaded_model = tf.keras.models.load_model(model_path)

def load_and_preprocess_images_from_directory(directory_path):
    image_list = []

    for filename in os.listdir(directory_path):
        if filename.endswith('.JPG'): 
            image_path = os.path.join(directory_path, filename)
            img = Image.open(image_path)

            img = img.resize((224, 224))

            img_array = np.array(img)

            img_array = img_array / 255.0

            image_list.append(img_array)

    return np.array(image_list)

def classify_images(model, test_data, class_names):
    print("Number of test images loaded:", len(test_data))

    predictions = model.predict(test_data)

    # Display the results
    for i, prediction in enumerate(predictions):
        class_index = prediction.argmax()
        predicted_class = class_names[class_index]

        print(f"Image {i + 1} is classified as: {predicted_class}")

test_directory = "E:\Minor\TASK\TEST"  

test_data = load_and_preprocess_images_from_directory(test_directory)

classify_images(loaded_model, test_data, class_names)
