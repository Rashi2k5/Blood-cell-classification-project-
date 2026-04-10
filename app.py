import os
import numpy as np
import cv2
from flask import Flask, request, render_template, redirect
import tensorflow as tf
import base64

app = Flask(__name__)

# --- MODEL LOADING ---
MODEL_PATH = "blood_cell_classifier.h5"

print("Loading model... please wait.")

try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Mubarak ho! Model load ho gaya.")
except Exception as e:
    print(f"Loading error: {e}")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects={'InputLayer': tf.keras.layers.InputLayer})

class_labels = ['eosinophil', 'lymphocyte', 'monocyte', 'neutrophil']

def predict_image_class(image_path, model):
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (224, 224))
    
    # Preprocessing
    img_array = np.array(img_resized, dtype="float32")
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)[0]
    
    # YAHAN FIX KIYA HAI: Label ko variable mein store kiya
    predicted_label = class_labels[predicted_class_idx]
    
    return predicted_label, img_rgb

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename != "":
            if not os.path.exists('static'):
                os.makedirs('static')
            file_path = os.path.join("static", file.filename)
            file.save(file_path)
            
            # Prediction calling
            label, img_rgb = predict_image_class(file_path, model)
            
            print(f"DEBUG: Prediction is {label}")
            # Image encoding for HTML
            _, img_encoded = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            img_str = base64.b64encode(img_encoded).decode('utf-8')
            
            return render_template("result.html", class_label=label, img_data=img_str)
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)