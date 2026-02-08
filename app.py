from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os


app = Flask(__name__)


# Load trained model
model = load_model("best_eye_model4.keras")


# Class labels
class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']


UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)




def predict_image(img_path):
    """Predict eye disease with confidence threshold to reject non-eye images."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)


    preds = model.predict(img_array)
    class_index = np.argmax(preds)
    confidence = float(np.max(preds))


# ⭐ Confidence threshold for non-eye rejection
    if confidence < 0.60:
        return "Nahi bata raha nikal", confidence
    
    
    return class_names[class_index], confidence




@app.route('/', methods=['GET', 'POST'])
def index():
    label = None
    confidence = None
    img_path = None


    if request.method == 'POST':
        file = request.files.get('file')


        if file and file.filename != "":
            img_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(img_path)


            label, confidence = predict_image(img_path)
    
    
    return render_template('index.html', label=label, confidence=confidence, img_path=img_path)




if __name__ == '__main__':
    app.run(debug=True)