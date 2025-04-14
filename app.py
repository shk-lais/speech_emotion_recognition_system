from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
import numpy as np
import librosa
import pickle
from keras.models import load_model

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load model and encoder
model = load_model("speech_emotion_recognition.keras", compile=False)
with open("label_encoder.pkl", "rb") as f:
    lb = pickle.load(f)

# MFCC extractor
def extract_mfcc(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    return mfcc

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None

    if request.method == 'POST':
        file = request.files['audio']
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            features = extract_mfcc(filepath).reshape(1, -1)
            pred = model.predict(features)
            prediction = lb.inverse_transform([np.argmax(pred)])[0]
            filename = file.filename

    return render_template('index.html', prediction=prediction, filename=filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(debug=True)
