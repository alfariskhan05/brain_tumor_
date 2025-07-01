from flask import Flask, render_template, request, redirect, url_for, session, flash
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
import numpy as np
import cv2

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load U-Net model
model = smp.Unet(encoder_name='densenet121', encoder_weights=None, in_channels=1, classes=1)
model.load_state_dict(torch.load('brain_tumor_segmentation.pt', map_location='cpu'))
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form['username'] == 'admin' and request.form['password'] == 'admin':
            session['user'] = 'admin'
            return redirect(url_for('upload'))
        else:
            flash('Invalid Credentials')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    return render_template('register.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if 'user' not in session:
        return redirect(url_for('login'))
    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image = Image.open(filepath).convert('RGB')
            input_tensor = transform(image).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
            mask = torch.sigmoid(output).squeeze().numpy()
            mask = (mask > 0.5).astype(np.uint8) * 255
            mask = cv2.resize(mask, image.size)

            overlay = np.array(image)
            overlay[mask > 0] = [0, 255, 0]
            result = cv2.addWeighted(np.array(image), 0.6, overlay, 0.4, 0)

            result_path = os.path.join(app.config['UPLOAD_FOLDER'], f"result_{filename}")
            Image.fromarray(result).save(result_path)

            return render_template('result.html', original=filepath, result=result_path)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
