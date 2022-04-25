import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# The decoder that is used to determine the classification output of the exterior classification model for a single image
def exterior_classification(img_path, saved_model_path):
    ext_model = tf.keras.models.load_model(saved_model_path)

    img = image.load_img(img_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_preprocessed = np.expand_dims(img_array, axis=0)

    predictions = ext_model(img_preprocessed)

    label = np.argmax(predictions, axis=1)[0]
    decoder = {0: 'acceptable_image', 1: 'not_acceptable_image'}
    decoded_label = decoder[label]
    return decoded_label