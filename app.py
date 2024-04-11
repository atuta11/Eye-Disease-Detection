import os
import numpy as np
from keras.models import load_model
from flask import Flask, request, render_template
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

# Load the trained model
model = load_model("my_model.h5")

# Resize the image to the expected size
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
@app.route('/index.html')
def index():
    
    # Assuming the image is named "home.jpg" and is located in the "static" folder
    image_path = "/static/home.jpg"
    return render_template('index.html', image_path=image_path)

   

@app.route('/result', methods=["POST"])
def result():
    if request.method == "POST":
        f = request.files['image']
        basepath = os.path.dirname(__file__)
        filepath = os.path.join(basepath, 'uploads', f.filename)
        f.save(filepath)

        # Load and preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make prediction
        prediction = model.predict(img_array)
        prediction_class = np.argmax(prediction)
        classes = {0: 'Bulging_Eyes', 1: 'Cataracts', 2: 'Crossed_Eyes', 3: 'Normal', 4: 'Uveitis'}
        result = classes[prediction_class]

        return render_template('result.html', pred=result)
        
# Route for the Cataract page
@app.route('/cataract')
def cataract():
    return render_template('cataract.html')

# Route for the Glaucoma page
@app.route('/glucoma')
def glucoma():
    return render_template('glucoma.html')

# Route for the Bulging Eyes page
@app.route('/bulging_eyes')
def bulging_eyes():
    return render_template('bulging_eyes.html')

# Route for the Crossed Eyes page
@app.route('/crossed_eyes')
def crossed_eyes():
    return render_template('crossed_eyes.html')

# Route for the Uveitis page
@app.route('/uveitis')
def uveitis():
    return render_template('uveitis.html')

# Route for the Eye Care page (assuming it's a placeholder)
@app.route('/eye_care')
def eye_care():
    return render_template('eye_care.html')



@app.route('/view_detail/<disease>', methods=["GET"])
def view_detail(disease):
    if disease == 'Cataracts':
        # Render cataract.html with specific data
        data = "This is the cataract page"
        return render_template('cataract.html', data=data)
    if disease == 'Crossed_Eyes':
        
        data = "This is the cataract page"
        return render_template('crossed_eyes.html', data=data)
    if disease == 'Bulging_Eyes':
        
        data = "This is the cataract page"
        return render_template('bulging_eyes.html', data=data)
    
    if disease == 'Uveitis':
        
        data = "This is the cataract page"
        return render_template('uveitis.html', data=data)

    # Add similar blocks for other diseases if needed
    else:
        # Handle other diseases or invalid routes
        return "Details not available for this disease."


if __name__ == "__main__":
    app.run(debug=True)