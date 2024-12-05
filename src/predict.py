from keras.models import load_model
from feature_extraction import extract_features
import numpy as np
from sklearn.preprocessing import LabelEncoder


# Load the trained model
model = load_model('vocalemotion.h5')

# Define label encoder
le = LabelEncoder()
le.fit(['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised'])

def predict_emotion(file_name):
    # Extract features from the new audio file
    features = extract_features(file_name)
    if features is not None:
        features = np.expand_dims(features, axis=0)  # Reshape for model input
        features = np.expand_dims(features, axis=-1) # Add channel dimension

        # Predict emotion
        prediction = model.predict(features)
        predicted_label = np.argmax(prediction)
        emotion = le.inverse_transform([predicted_label])[0]
        return emotion
    else:
        return "Error in feature extraction"

# Example usage
audio_file = "data\\TEST\\angry.wav"
emotion = predict_emotion(audio_file)
print(f"The predicted emotion is: {emotion}")

audio_file = "data\\TEST\\calm.wav"
emotion = predict_emotion(audio_file)
print(f"The predicted emotion is: {emotion}")

audio_file = "data\\TEST\\disgust.wav"
emotion = predict_emotion(audio_file)
print(f"The predicted emotion is: {emotion}")
