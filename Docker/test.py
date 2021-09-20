import tensorflow_datasets as tfds
import tensorflow as tf

#Load model
model_path = '/tmp/keras-model'
loaded_model = tf.keras.models.load_model(model_path)

#Load dataset
test_ds, ds_info  = tfds.load('cifar10', split='test', shuffle_files=True, with_info=True)

#Dataset preparation
size = test_ds.cardinality().numpy()
test_images, test_labels = [None] * size, [None] * size
for i, data in enumerate(test_ds.take(size)):
    test_images[i] = data["image"]
    test_labels[i] = data["label"]
test_ds=tf.data.Dataset.from_tensor_slices((test_images, test_labels))

test_ds = (test_ds
                  .map(lambda x, y: (tf.image.per_image_standardization(x), y))
                  .shuffle(buffer_size=10000)
                  .batch(batch_size=40, drop_remainder=True))

#Get result
loaded_model.evaluate(test_ds)
