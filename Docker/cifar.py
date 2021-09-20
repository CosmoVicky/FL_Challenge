import os
import tensorflow as tf
import numpy as np
from tensorflow import keras

def dataset(batch_size):
  (images, labels), _ = tf.keras.datasets.cifar10.load_data()
  # let's use 20% of train sample for validation
  sub_size = int(images.shape[0] * 0.2)

  validation_images, validation_labels = images[:sub_size], labels[:sub_size]
  train_images, train_labels = images[sub_size:], labels[sub_size:]

  train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
  validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

  train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
  val_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()

  train_ds = (train_ds
                  .map(lambda x, y: (tf.image.per_image_standardization(x), y))
                  .shuffle(buffer_size=train_ds_size)
                  .batch(batch_size=batch_size, drop_remainder=True))
  validation_ds = (validation_ds
                  .map(lambda x, y: (tf.image.per_image_standardization(x), y))
                  .shuffle(buffer_size=val_ds_size)
                  .batch(batch_size=batch_size, drop_remainder=True))
  return [train_ds, validation_ds]

def build_and_compile_model():
  model = keras.models.Sequential([
    keras.layers.Conv2D(filters=96, kernel_size=(5,5), strides=(2,2), activation='relu', input_shape=(32,32,3)),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(1,1)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3)),
    keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(512,activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10,activation=tf.keras.activations.softmax)])
  
  model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(
    learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
    name='Adam'), metrics=['accuracy'])
  model.summary()
  return model
