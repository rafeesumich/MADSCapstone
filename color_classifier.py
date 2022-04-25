import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib
import matplotlib.pyplot as plt

data_dir = pathlib.Path('cars_by_color/')  # Sets path to dataset for model
image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)


batch_size = 16

# Sets dimensions to resize the image to
img_height = 180
img_width = 180

# Create the train dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='training',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size)

# Create the validation dataset
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset='validation',
    seed=42,
    image_size=(img_height, img_width),
    batch_size=batch_size)

print(train_ds.class_names)
class_names = train_ds.class_names

# Creates test set as a portion of the validation set and removes the test set portion from the validation set
test_size = val_ds.cardinality()//2
test_dataset = val_ds.take(test_size)
val_ds = val_ds.skip(test_size)

print('Batches for testing -->', test_dataset.cardinality())
print('Batches for validating -->', val_ds.cardinality())

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(100).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = len(class_names)

# Create model architecture
model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(.2),
  layers.Dense(num_classes, activation='softmax')
])


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.1, restore_best_weights=True)]

print(model.summary())

epochs = 10

# Fits model to train data
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=callbacks
)


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# Creates line plots over epochs for training and validation accuracy
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Creates line plots over epochs for training and validation loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

result = model.evaluate(test_dataset)
print(dict(zip(model.metrics_names, result)))  # shows test accuracy and loss

#model.save('saved_color_model')