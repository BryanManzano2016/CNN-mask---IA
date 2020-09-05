import json
import os
from contextlib import redirect_stdout
from datetime import datetime

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

params = {
    "epochs": 20,
    "batch_size": 32,
    "target_dir": './model_project_' + datetime.today().strftime('%Y%m%d_%H%M%S') + "/",
    "name_dir_dataset": "dataset",
    "size_image": 200,
    "num_classes": 2,
    "validation_split": 0.2,
    "seed": 123,
    "rotation": 0.2,
    "zoom": 0.2,
    "drop_out": 0.3,
    "random_flip": "horizontal_and_vertical"
}

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    params.get("name_dir_dataset"),
    validation_split=params.get("validation_split"),
    subset="training",
    seed=params.get("seed"),
    image_size=(params.get("size_image"), params.get("size_image")),
    batch_size=params.get("batch_size"))

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    params.get("name_dir_dataset"),
    validation_split=params.get("validation_split"),
    subset="validation",
    seed=params.get("seed"),
    image_size=(params.get("size_image"), params.get("size_image")),
    batch_size=params.get("batch_size"))

data_augmentation = tf.keras.Sequential([
    layers.experimental.preprocessing.RandomFlip(params.get("random_flip"), input_shape=(params.get("size_image"),
                                                                                         params.get("size_image"),
                                                                                         3)),
    layers.experimental.preprocessing.RandomRotation(params.get("rotation")),
    layers.experimental.preprocessing.RandomZoom(params.get("zoom")),
])

model = Sequential([
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1. / 255),
    layers.Conv2D(64, 5, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(256, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    Dropout(params.get("drop_out")),
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(params.get("num_classes"))
])

model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=params.get("epochs")
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(params.get("epochs"))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')

if not os.path.exists(params.get("target_dir")):
    os.mkdir(params.get("target_dir"))

model.save(params.get("target_dir") + 'modelo.h5')
model.save_weights(params.get("target_dir") + 'pesos.h5')

plt.savefig(params.get("target_dir") + '/plot.png', bbox_inches='tight')

json_text = json.dumps(params)

with open(params.get("target_dir") + '/params.txt', 'w') as f:
    f.write(json_text)

with open(params.get("target_dir") + '/model.txt', 'w') as f:
    with redirect_stdout(f):
        model.summary()
