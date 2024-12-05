from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from model import create_model

# Load preprocessed data
X_train = np.load('data\\X_train.npy')  # Corrected file path
y_train = np.load('data\\y_train.npy')
X_test = np.load('data\\X_test.npy')
y_test = np.load('data\\y_test.npy')

# Encode labels (integer encoding)
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Check the size of the datasets
print(f"X_train size: {X_train.shape[0]}, y_train_encoded size: {len(y_train_encoded)}")
print(f"X_test size: {X_test.shape[0]}, y_test_encoded size: {len(y_test_encoded)}")

# Reshape X data to be 3D (samples, timesteps, features) for Conv1D
X_train = np.expand_dims(X_train, axis=-1)
X_test = np.expand_dims(X_test, axis=-1)

# Model architecture
model = create_model((X_train.shape[1], 1))

# Compile the model with sparse_categorical_crossentropy loss
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Initialize and train the model
history = model.fit(X_train, y_train_encoded, epochs=50, batch_size=32, validation_data=(X_test, y_test_encoded))

# Save the trained model
model.save('vocalemotion.h5')
print("Model training complete! Model saved.")
