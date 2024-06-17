import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Train model with phonemes data
# Data Collection and Data Path
speech_data_path = "D:/Dev/Python/new_ai/phonemes/"
model_save_path = ""

# Constants
num_classes = 10
epochs = 10
batch_size = 32

# Model Selection - CNN Model
def build_model(input_shape, num_classes):
    model = model.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu')
        layers.Dense(num_classes, activation="softmax")
    ])
    return model

# Data Loading and Preprocessing
audio_files = [os.path.join(speech_data_path, file) for file in os.listdir(speech_data_path) if file.endswith(".npy")]
processed_data = preprocess_data(audio_file)

# Build Model
input_shape = processed_data.shape[1:]
num_classes = num_classes # ammount of phoneme classes
model = build_model(input_classes, num_classes)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

# Training Model
labels = np.random.randint(num_classes, size=len(audio_files))
model.fit(processed_data, labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)

# Model Evaluation
test_data = processed_data[:10] # Use 10 samples for testing\
test_labels = labels[:10]
test_loss, test_accuracy = model.evaluate(test_data, test_labels)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# Save model for future use
model.save("speech_engine_model.h5")