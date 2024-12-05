import os
import shutil

# Define a dictionary to map the emotion codes to labels
emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

# Define the source directory where the RAVDESS audio files are located
source_dir = "data\Audio_Speech_Actors_01-24"

# Define the destination directory where you want to organize the files by emotion
dest_dir = "data/Emotions_Sorted"

# Create directories for each emotion
def create_emotion_directories():
    for emotion in emotion_map.values():
        emotion_dir = os.path.join(dest_dir, emotion)
        if not os.path.exists(emotion_dir):
            os.makedirs(emotion_dir)
            print(f"Created directory: {emotion_dir}")

# Function to get emotion label from the filename
def get_emotion_label_from_filename(file_name):
    parts = file_name.split("-")
    emotion_code = parts[2]  # The third part represents the emotion code
    emotion = emotion_map.get(emotion_code, "unknown")  # Map it to the emotion label
    return emotion

# Function to move files into emotion folders
def organize_files_by_emotion(data_dir):
    for actor_folder in os.listdir(data_dir):
        actor_path = os.path.join(data_dir, actor_folder)
        if os.path.isdir(actor_path):
            for file in os.listdir(actor_path):
                if file.endswith(".wav"):
                    file_path = os.path.join(actor_path, file)
                    emotion = get_emotion_label_from_filename(file)
                    if emotion != "unknown":  # Only move known emotions
                        dest_path = os.path.join(dest_dir, emotion, file)
                        print(f"Moving {file} to {emotion} folder")
                        shutil.move(file_path, dest_path)  # Move the file to the correct folder

# Create the directories first
create_emotion_directories()

# Organize the files into their respective emotion folders
organize_files_by_emotion(source_dir)

print("Files organized by emotion!")
