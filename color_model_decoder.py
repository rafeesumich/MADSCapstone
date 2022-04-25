import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# The decoder that is used to determine the classification output of the color classification model for a single image
def color_classification(img_path,model):

    img = image.load_img(img_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_preprocessed = np.expand_dims(img_array, axis=0)

    predictions = model(img_preprocessed)

    label = np.argmax(predictions, axis=1)[0]
    decoder = {0:'Beige', 1:'Black', 2:'Blue', 3:'Brown', 4:'Gold', 5:'Gray', 6:'Green', 7:'Orange', 8:'Pink', 9:'Purple', 10:'Red', 11:'Silver', 12:'White', 13:'Yellow'}
    decoded_label = decoder[label]
    return decoded_label

saved_model_path = ''
img_path = ''

model = tf.keras.models.load_model(saved_model_path)
print(color_classification(img_path, model))
