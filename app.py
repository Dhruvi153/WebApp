import os
from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("models/final.h5")

# Define class labels
class_labels = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

# Configure a folder for uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to preprocess and classify an image
def classify_image(file_path):
    # Load and preprocess the image
    img = Image.open(file_path)
    img = img.resize((224, 224))
    img_array = np.array(img) 
    if img_array.shape[2] == 4:  # If image has an alpha channel, remove it
        img_array = img_array[:, :, :3]

    # Ensure uint8 data type
    img_array = img_array.astype(np.uint8)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)

    # Thresholding and adjustment
    threshold = 0.37757874193797547
    y_test_thresholded = predictions > threshold
    y_test_binary = y_test_thresholded.astype(int)
    y_test_adjusted = y_test_binary.sum(axis=1) - 1
      
    if y_test_adjusted == 0:
        return "No DR"
    elif y_test_adjusted == 1:
        return "Mild"
    elif y_test_adjusted == 2:
        return "Moderate"
    elif y_test_adjusted == 3:
        return "Severe"
    elif y_test_adjusted == 4:
        return "Proliferative"

    
    
    #return y_test_adjusted

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('home.html', error='No file part')
        
        file = request.files['file']
        
        # Check if the file has a valid name and extension
        if file.filename == '':
            return render_template('home.html', error='No selected file')
        
        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            
            # Classify the uploaded image
            y_test_adjusted = classify_image(file_path)
            
            return render_template('prediction.html', 
                                   image_path=file_path, 
                                   predicted_class=y_test_adjusted, 
                                   )
    
    return render_template('home.html', error=None)

if __name__ == '__main__':
    app.run(debug=True)
