import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define data directories and parameters
train_data_dir = "E:\Minor\TASK_2B\TRAIN"
validation_data_dir = "E:\Minor\TASK_2B\TEST"
input_shape = (224, 224)  # Adjust the input shape as per your dataset
batch_size = 32
num_epochs = 100

# Create data generators for data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,  # Normalize pixel values
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=True,
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=input_shape,
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False,
)

# Define the model architecture with more hidden layers
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(input_shape[0], input_shape[1], 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.5),  # Added dropout layer
    layers.Dense(128, activation="relu"),
    layers.Dense(5, activation="softmax"),  # Replace num_classes with the number of classes in your dataset
])

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
)

# Save the trained model
model.save("handwriting_recognition_model2.h5")