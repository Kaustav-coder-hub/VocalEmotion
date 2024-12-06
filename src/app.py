from flask import Flask, render_template, request, redirect, url_for
from keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import sequence
import librosa
import os

app = Flask(__name__)

# Load the model
model = load_model('vocalemotion.h5')

# Define the allowed audio file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'flac'}

# Check if file is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function to preprocess the audio file
def preprocess_audio(file_path):
    # Load the audio file
    audio, sr = librosa.load(file_path, sr=None)

    # Example of extracting MFCC features, you can adjust this depending on your model's requirements
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=64)  # Extract 64 MFCC coefficients
    mfcc = np.mean(mfcc.T, axis=0)  # Average across time axis, resulting in a 64-dimensional feature vector

    # Reshape or pad the data to ensure it matches the expected input shape
    # Your model expects input of shape (batch_size, sequence_length, num_features)
    # Here we reshape it to (1, 40, 64), because the model uses a sequence length of 40
    mfcc = np.reshape(mfcc, (1, 40, 64))  # Ensure the shape is (1, 40, 64)

    return mfcc

# Route to the home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle the audio file upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file and allowed_file(file.filename):
        # Ensure the uploads directory exists
        upload_folder = 'uploads'
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)  # Create the folder if it doesn't exist
        
        filename = os.path.join(upload_folder, file.filename)
        file.save(filename)
        model.summary()
        
        # Preprocess and make a prediction
        audio_data = preprocess_audio(filename)
        prediction = model.predict(audio_data)
        
        # Get the predicted emotion (assuming 6 classes)
        predicted_class = np.argmax(prediction)
        emotions = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        predicted_emotion = emotions[predicted_class]
        
        return render_template('result.html', emotion=predicted_emotion)
    
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
