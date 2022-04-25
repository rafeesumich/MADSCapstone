import tensorflow as tf
from os import listdir
from os.path import isfile, join
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil
import os

ext_model = tf.keras.models.load_model('acceptable_image_classifier')  # Loads acceptable image classifier

all_images_folder = 'all_car_images/'

# Get all names of subdirectories in directory
folders_in_all_images_folder = [os.path.join(all_images_folder, file) for file in os.listdir(all_images_folder) if os.path.isdir(os.path.join(all_images_folder, file))]

# Iterates though subdirectories to get all image files
for folder in folders_in_all_images_folder[1:]:
    print(folder)
    folder_num = folder[32:]
    ext_filtered_folder = 'ext_filtered_imgs_' + folder_num

    target_folder = 'exterior_filtered_images/' + ext_filtered_folder
    os.mkdir(target_folder)

    new_exterior_folder = target_folder + '/exterior'
    os.mkdir(new_exterior_folder)

    new_misc_folder = target_folder + '/misc'
    os.mkdir(new_misc_folder)

    files_in_folder = [f for f in listdir(folder) if isfile(join(folder, f))]  # Makes list of image files

    # Iterates through image files and sorts them into directories based on whether they are acceptable (exterior) or not acceptable (misc.) images
    for file_name in files_in_folder:

        full_path = folder + '/' + file_name

        try:
            img = image.load_img(full_path, target_size=(180, 180))
            img_array = image.img_to_array(img)
            img_preprocessed = np.expand_dims(img_array, axis=0)

            predictions = ext_model(img_preprocessed)

            label = np.argmax(predictions, axis=1)[0]
            decoder = {0: 'acceptable_image', 1: 'not_acceptable_image'}
            decoded_label = decoder[label]

            if decoded_label == 'acceptable_image':
                shutil.move(full_path, new_exterior_folder)
            else:
                shutil.move(full_path, new_misc_folder)
        except:
            print('could not load image: ' + full_path)
