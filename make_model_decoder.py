import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# The decoder that is used to determine the classification output of the make classification model for a single image

def make_classification(img_path,model):

    img = image.load_img(img_path, target_size=(220, 220))
    img_array = image.img_to_array(img)
    img_preprocessed = np.expand_dims(img_array, axis=0)

    predictions = model(img_preprocessed)

    label = np.argmax(predictions, axis=1)[0]
    decoder = {0: 'Acura', 1: 'Alfa Romeo', 2: 'Aston Martin', 3: 'Audi',
               4: 'BMW', 5: 'Bentley', 6: 'Buick', 7: 'Cadillac', 8: 'Chevrolet',
               9: 'Chrysler', 10: 'Dodge', 11: 'FIAT', 12: 'Ferrari', 13: 'Ford',
               14: 'GENESIS', 15: 'GMC', 16: 'Honda', 17: 'Hummer', 18: 'Hyundai',
               19: 'INFINITI', 20: 'Isuzu', 21: 'Jaguar', 22: 'Jeep', 23: 'KARMA',
               24: 'Kia', 25: 'Lamborghini', 26: 'Land Rover', 27: 'Lexus', 28: 'Lincoln',
               29: 'Lotus', 30: 'MINI', 31: 'Maserati', 32: 'Maybach', 33: 'Mazda',
               34: 'McLaren', 35: 'Mercedes-Benz', 36: 'Mercury', 37: 'Mitsubishi',
               38: 'Nissan', 39: 'Oldsmobile', 40: 'Plymouth', 41: 'Pontiac',
               42: 'Porsche', 43: 'RAM', 44: 'Rolls-Royce', 45: 'Saab',
               46: 'Saturn', 47: 'Scion', 48: 'Subaru', 49: 'Suzuki', 50: 'Toyota',
               51: 'Volkswagen', 52: 'Volvo', 53: 'smart'}
    decoded_label = decoder[label]
    return decoded_label

saved_model_path = ''
img_path = ''

model = tf.keras.models.load_model(saved_model_path)
print(make_classification(img_path, model))
