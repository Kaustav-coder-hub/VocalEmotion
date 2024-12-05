# VocalEmotion
VocalEmotion is a speech emotion recognition model that leverages deep learning techniques to classify emotions from audio files. This project focuses on various emotional states such as happiness, anger, sadness, and more, using the RAVDESS dataset.

# Table of Contents

1. [Features](https://github.com/Kaustav-coder-hub/VocalEmotion/edit/master/Readme.md#features)
2. [Installation](https://github.com/Kaustav-coder-hub/VocalEmotion/edit/master/Readme.md#Installation)
3. [Usage](https://github.com/Kaustav-coder-hub/VocalEmotion/edit/master/Readme.md#Usage)
4. [Dataset](https://github.com/Kaustav-coder-hub/VocalEmotion/edit/master/Readme.md#Dataset)
5. [Project Structure](https://github.com/Kaustav-coder-hub/VocalEmotion/edit/master/Readme.md#Project-Structure)
6. [License](https://github.com/Kaustav-coder-hub/VocalEmotion/edit/master/Readme.md#License)
7. [Acknowledgments](https://github.com/Kaustav-coder-hub/VocalEmotion/edit/master/Readme.md#Acknowledgments)

# Features
1. Emotion classification from audio files using deep learning.
2. Organized dataset with folders for each emotion.
3. Built using Keras and TensorFlow for efficient training and inference.

# Installation

1. Clone the repository:
     ```
     (https://github.com/Kaustav-coder-hub/VocalEmotion)
     ```

2. Navigate to the project directory:
    ```
    cd VocalEmotion
    ```

3. Create and activate a virtual environment (optional but recommended):
    ```
    python -m venv vocalemotion-env
    
    source vocalemotion-env/bin/activate  # On Windows, use vocalemotion-env\Scripts\activate
    ```

4. Install the required dependencies:
    ```
    pip install -r requirements.txt
    ```

# Usage

1. **Organize the Data:** Run the `organize.py` script to organize audio files into emotion-specific folders.
     ```
    python src/organize.py
    ```
    
3. **Preprocess the Data:** Run the preprocessing script to extract features from the organized audio files.
    ```
    python src/preprocess_data.py
    ```

4. **Train the Model:** Train the emotion recognition model.
   ```
   python src/train.py
   ```
5. **Make Predictions:** Use the trained model to predict emotions from new audio files.
    ```
    python src/predict.py
    ```
# Dataset
  The project uses the RAVDESS dataset, which contains emotional speech recordings. The dataset can be downloaded from here.

Directory Structure
1. **Raw Audio Data:** The dataset is initially structured into actor-specific folders under the Audio_Speech_Actors_01-24 directory.
    ```
    data/
    ├── Audio_Speech_Actors_01-24/
    │   ├── Actor_01/
    │   ├── Actor_02/
    │   ├── Actor_03/
    │   ├── Actor_04/
    │   └── ... (until Actor_24)
    ```
2. **Organized by Emotion:** After running the organize.py script, the audio files are organized into emotion-specific folders under the emotions_sorted directory.
    ```
    data/
    ├── emotions_sorted/
    │   ├── happy/
    │   ├── sad/
    │   ├── angry/
    │   ├── fearful/
    │   ├── calm/
    │   ├── neutral/
    │   ├── disgust/
    │   └── surprised/
    ```
  The `organize.py` script takes the audio files from the Audio_Speech_Actors_01-24 directory and organizes them into folders based on the emotion encoded in the filename.

# Project Structure

    VocalEmotion/
    │
    ├── data/                   # Directory containing audio data
    │   ├── Audio_Speech_Actors_01-24/  # Raw audio files organized by actors
    │   ├── emotions_sorted/    # Organized by emotion
    │
    ├── src/                    # Source code for the project
    │   ├── preprocess_data.py   # Preprocessing script
    │   ├── train.py             # Training script
    │   ├── predict.py           # Prediction script
    │   ├── feature_extraction.py # Feature extraction functions
    │   └── organize.py          # Script to organize audio files by emotion
    │
    ├── requirements.txt        # Python dependencies
    ├── README.md               # Project documentation
    └── vocalemotion_model.h5   # Trained model file
    
    
# License
  This project is licensed under the MIT License. See the [LICENSE](https://github.com/Kaustav-coder-hub/VocalEmotion/blob/master/LICENSE) file for details.

# Acknowledgments
1. RAVDESS dataset creators for providing the emotional speech recordings.
2. Keras and TensorFlow for the deep learning framework.