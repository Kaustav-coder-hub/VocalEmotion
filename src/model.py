from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten

def create_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, kernel_size=5, strides=1, padding="same", activation="relu", input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(8, activation='softmax'))  # Adjust based on the number of emotion classes
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
