import os
import cv2
import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
IMG_SIZE = 128
LABELS = ['PNEUMONIA', 'NORMAL']

def prepare_image(filepath):
    # Grayscale read (as per your notebook)
    img_arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    resized_arr = cv2.resize(img_arr, (IMG_SIZE, IMG_SIZE))
    
    # Convert to 3-channel (RGB) by stacking because pre-trained models require 3 channels
    img_rgb = cv2.merge([resized_arr, resized_arr, resized_arr])
    
    normalized_img = img_rgb / 255.0 
    return normalized_img.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', message='No file uploaded')
    
    file = request.files['file']
    # Get the user's wish for which model to use
    selected_model_name = request.form.get('model_choice')
    
    if file.filename != '':
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Construct the filename based on choice (e.g., "resnet50_model.h5")
        model_path = f"{selected_model_name}_model.h5"
        
        if not os.path.exists(model_path):
            return render_template('index.html', message=f"Error: {model_path} not found in folder!")

        # LOAD ON DEMAND: This keeps your RAM free
        model = load_model(model_path)
        
        # Predict
        processed_img = prepare_image(filepath)
        prediction = model.predict(processed_img)
        
        result = LABELS[np.argmax(prediction)]
        confidence = round(100 * np.max(prediction), 2)

        return render_template('index.html', 
                               prediction=result, 
                               confidence=confidence, 
                               image_path=filepath,
                               model_used=selected_model_name)

    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)