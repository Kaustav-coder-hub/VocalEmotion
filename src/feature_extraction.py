import librosa
import numpy as np

def extract_features(file_name):
    try:
        audio, sample_rate = librosa.load(file_name, sr=None, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        return mfccs_mean
    except Exception as e:
        print(f"Error encountered while parsing file: {file_name}")
        return None
