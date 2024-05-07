import numpy as np
from PIL import Image
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import os


app = Flask(__name__, static_url_path='/static')

class_names = ["Aditi", "Kratika", "Radhika", "Shreya", "Sahitya"]
model_path = "handwriting_recognition_model2.h5"
loaded_model = tf.keras.models.load_model(model_path)

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = img_array / 255.0
    return img_array

def classify_image(image, model, class_names):
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0]
    class_index = np.argmax(prediction)
    predicted_class = class_names[class_index]
    return predicted_class

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        img_array = preprocess_image(file)
        predicted_class = classify_image(img_array, loaded_model, class_names)
        return jsonify({'predicted_class': predicted_class})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(port=int(os.environ.get("PORT", 8080)), host='0.0.0.0', debug=True)
