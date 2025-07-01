import os
import zipfile
import uuid
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import segmentation_models_pytorch as smp
from torchvision import transforms

app = Flask(__name__)
app.secret_key = 'supersecret'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

users = {}

class User(UserMixin):
    def __init__(self, id):
        self.id = id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id) if user_id in users else None

# 🧠 Unzip the model if not already extracted
model_path = 'models/brain_tumor_segmentation.pt'
if not os.path.exists(model_path):
    with zipfile.ZipFile('models/brain_tumor_segmentation.zip', 'r') as zip_ref:
        zip_ref.extractall('models')

# 🔁 Load Model
model = smp.Unet(encoder_name="densenet121", in_channels=3, classes=1)
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        users['user'] = {'password': 'pass'}
        login_user(User('user'))
        return redirect(url_for('upload'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        file = request.files['file']
        filename = str(uuid.uuid4()) + "_" + file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        input_tensor = preprocess_image(filepath)
        with torch.no_grad():
            output = model(input_tensor)
        result = (output.squeeze().numpy() > 0.5).astype(np.uint8) * 255
        overlay_path = os.path.join(app.config['UPLOAD_FOLDER'], "overlay_" + filename)
        cv2.imwrite(overlay_path, result)
        return render_template('result.html', original=filename, overlay="overlay_" + filename)
    return render_template('upload.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
