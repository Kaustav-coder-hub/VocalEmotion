import os
import numpy as np
from feature_extraction import extract_features
from sklearn.model_selection import train_test_split

# Paths to the data folders
data_dir = "data/Emotions_Sorted"

# Lists to store features and labels
features = []
labels = []

# Define a function to process each dataset folder
def process_data(data_dir):
    for emotion in os.listdir(data_dir):
        emotion_path = os.path.join(data_dir, emotion)
        if os.path.isdir(emotion_path):
            for file in os.listdir(emotion_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(emotion_path, file)
                    feature = extract_features(file_path)  # Extract features from audio file
                    if feature is not None:  # Check if feature extraction was successful
                        features.append(feature)
                        labels.append(emotion)  # Use the folder name as the label


# Process both datasets (use different labels for speech and song for now)
process_data(data_dir)

# Convert to numpy arrays
X = np.array(features)
y = np.array(labels)

# Assert to check if features and labels match
assert len(features) == len(labels), f"Features size: {len(features)} and Labels size: {len(labels)} are not equal"

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Print out the sizes of X_train, X_test, y_train, and y_test
print(f"X_train size: {len(X_train)}, y_train size: {len(y_train)}")
print(f"X_test size: {len(X_test)}, y_test size: {len(y_test)}")

# Optionally, save the preprocessed data if needed
np.save("data/X_train.npy", X_train)
np.save("data/y_train.npy", y_train)
np.save("data/X_test.npy", X_test)
np.save("data/y_test.npy", y_test)

print("Data preprocessing complete! Files saved.")
