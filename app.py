"""This app file includes 4 CNN image classifier and 1 Random Forest Regressor model to estimate the vehicle price. 
Streamlit library was used to create input features at front end."""




import numpy as np
import pickle
import pandas as pd
import streamlit as st 
import joblib
import sys
import os
from PIL import Image
sys.modules['Image'] = Image 
import tensorflow as tf
from tensorflow.keras.preprocessing import image

from dropdict import *



# import pretrained and saved exterior classifier model
exterior_model_clf = tf.keras.models.load_model('./packaged_exterior_classifier/saved_model_exterior_class')

# import pretrained and saved car make classifier model
car_make_clf = tf.keras.models.load_model('./packaged_make_classifier/saved_model_make_class')

# import pretrained and saved body type classifier model
bodytype_model_clf = tf.keras.models.load_model('./packaged_body_classifier/saved_model_body_class')

# import pretrained and saved color classifier model
color_model_clf = tf.keras.models.load_model('./packaged_color_classifier/saved_model_color_class')

# import pretrained and saved RandomForest Model pipeline
rf_pipe = joblib.load('oe_rf_depth20_50Estimators.sav')



def exterior_classification(img_path):
    '''This function returns the output from exterior classifier model if the image of a car given by user is Acceptable or Not Acceptable'''

    img = image.load_img(img_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_preprocessed = np.expand_dims(img_array, axis=0)

    predictions = exterior_model_clf(img_preprocessed)

    label = np.argmax(predictions, axis=1)[0]
    decoder = {0: 'Acceptable image', 1: 'Not acceptable image'}
    decoded_label = decoder[label]
    return decoded_label




def make_classification(img_path):
    '''This function returns the output from make classifier model'''
    img = image.load_img(img_path, target_size=(220, 220))
    img_array = image.img_to_array(img)
    img_preprocessed = np.expand_dims(img_array, axis=0)

    predictions = car_make_clf(img_preprocessed)

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




def body_classification(img_path):
    '''This function returns the output from body classifier model'''
    img = image.load_img(img_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_preprocessed = np.expand_dims(img_array, axis=0)

    predictions = bodytype_model_clf(img_preprocessed)

    label = np.argmax(predictions, axis=1)[0]
    decoder = {0:'Cargo Van', 1:'Convertible', 2:'Coupe', 3:'Crossover', 4:'Hatchback', 5:'Minivan', 6:'Pickup', 7:'Roadster', 8:'SUV', 9:'Sedan', 10:'Wagon'}
    decoded_label = decoder[label]
    return decoded_label



def color_classification(img_path):
    '''This function returns the output from color classifier model'''
    img = image.load_img(img_path, target_size=(180, 180))
    img_array = image.img_to_array(img)
    img_preprocessed = np.expand_dims(img_array, axis=0)

    predictions = color_model_clf(img_preprocessed)

    label = np.argmax(predictions, axis=1)[0]
    decoder = {0:'Beige', 1:'Black', 2:'Blue', 3:'Brown', 4:'Gold', 5:'Gray', 6:'Green', 7:'Orange', 8:'Pink', 9:'Purple', 10:'Red', 11:'Silver', 12:'White', 13:'Yellow'}
    decoded_label = decoder[label]
    return decoded_label


def car_price_predictor(miles, year, make, model, trim, vehicle_type, body_type, drivetrain, fuel_type, engine_block, engine_size, transmission, doors, cylinders, 
        city_mpg, highway_mpg, base_exterior_color, base_interior_color, is_certified, state, carfax_1_owner, carfax_clean_title):
    '''This function returns the Estimated value of the car and min,max price range from the Random Forest Regressor model.'''


    single_df = pd.DataFrame(
        
       columns=['miles', 'year', 'make', 'model', 'trim', 'vehicle_type','body_type', 'drivetrain', 'fuel_type', 'engine_block', 'engine_size', 'transmission', 'doors', 
       'cylinders', 'city_mpg', 'highway_mpg','base_exterior_color', 'base_interior_color', 'is_certified', 'state','carfax_1_owner', 'carfax_clean_title'], 
       
       data = np.array([miles, year, make, model, trim, vehicle_type, body_type, drivetrain, fuel_type, engine_block, engine_size, transmission, doors, cylinders, 
        city_mpg, highway_mpg, base_exterior_color, base_interior_color, is_certified, state, carfax_1_owner, carfax_clean_title]).reshape(1, 22))
   
    estimated_value = rf_pipe.predict(single_df)
    x_treanformed=rf_pipe['preprocessor'].transform(single_df)
    rf_estimators = rf_pipe['estimator'].estimators_

    preds=[]
    for estimator in rf_estimators:
        preds.append(estimator.predict(x_treanformed))
        
      
    data = [x[0] for x in preds]
    
    price_range = (min(data),max(data))
    return (estimated_value[0], price_range)



def main():

    '''This main function is for web interface development. Each input section is an input variable for the Random Forest Regressor model. '''
    st.set_page_config(layout="wide")
    html_temp = """
    <h1 style='text-align: center; color: black;'>AUTOMATES</h1>
    <div style="background-color:#FFCB05;padding:10px">
    <h1 style="color:#00274C;text-align:center;">Car Price Prediction </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

    #Column 1 is for required fields, Column 2 and Column 3 are for optional fields
    col1,col2,col3 = st.columns(3)
    with col1:
        st.header("Required fields")

        # upload section for exterior image of a car
        uploaded_file = st.file_uploader("Please upload an exterior image (.jpeg) of your car")

        # load and read uploaded file
        if uploaded_file is not None:

            with open(uploaded_file.name,"wb") as f:
                f.write(uploaded_file.getbuffer())

            #call exterior classification model to see if it is an acceptable or not acceptable image.
            message = exterior_classification(uploaded_file.name)
            
            #if image is a Not acceptable image, ask user to provide another exterior image.
            if message == 'Not acceptable image':
                st.text("Please upload an exterior image (.jpeg) of your car")
                # display message
                st.text(message)
                 # deleting uploaded saved picture after prediction
                os.remove(uploaded_file.name)
            else:
                # if the image is a Accepted image, call other classifiers and display outouts.
                color_msg = color_classification(uploaded_file.name)
                make_msg = make_classification(uploaded_file.name)
                body_msg = body_classification(uploaded_file.name)
                # deleting uploaded saved picture after prediction
                os.remove(uploaded_file.name)
                st.text(message)
                st.text(make_msg)
                st.text(body_msg)
                st.text(color_msg)


        # select car make
        make = st.selectbox("Make",car_make)

        #imputing the Car Make from classifier
        if uploaded_file is not None:
            if message == 'Acceptable image':
                if make == 'No Selection':
                    make = make_msg
 
        # select car model
        model = st.selectbox("Model",car_make_models[make])
        # select car trim
        trim = st.selectbox("Trim",car_make_model_trims[make][model])
        # input car milage
        miles = st.number_input('Miles',min_value=0, step=1)
        # select car model year
        year = st.selectbox("Year",car_year[make][model][trim])


    with col2:  
        st.header("Optional fields") 
        #select vehicle type
        vehicle_type = st.selectbox("Vehicle Type",car_vehicle_type)

        if make !='Make Selection' and model !='Make Selection' and trim !='Make Selection' and year !='Make Selection':
            if vehicle_type == 'No Selection':
                vehicle_type = -1

        # select body_type
        body_type = st.selectbox("Body Type",car_body_type)
        if make !='Make Selection' and model !='Make Selection' and trim !='Make Selection' and year !='Make Selection':
            if body_type == 'No Selection':
                body_type = -1


        #imputing the Body type from classifier
        if uploaded_file is not None:
            if message == 'Acceptable image':
                if body_type == 'No Selection':
                    body_type = body_msg

        # select car drivetrain type
        drivetrain = st.selectbox("Drive Type",car_drivetrain)

        if make !='Make Selection' and model !='Make Selection' and trim !='Make Selection' and year !='Make Selection':
            if drivetrain == 'No Selection':
                drivetrain = -1

        # select car transmission type
        transmission = st.selectbox("Transmission",car_transmission)

        if make !='Make Selection' and model !='Make Selection' and trim !='Make Selection' and year !='Make Selection':
            if transmission == 'No Selection':
                transmission = -1

        # select car fuel_type 
        fuel_type = st.selectbox("Fuel Type",car_fueltype)

        if make !='Make Selection' and model !='Make Selection' and trim !='Make Selection' and year !='Make Selection':
            if fuel_type == 'No Selection':
                fuel_type = -1

        # select engine block
        engine_block = st.selectbox("Engine Type",car_enginetype)

        if make !='Make Selection' and model !='Make Selection' and trim !='Make Selection' and year !='Make Selection':
            if engine_block == 'No Selection':
                engine_block = -1

        # select engine size
        engine_size = st.selectbox("Engine Size",car_enginesize)

        if make !='Make Selection' and model !='Make Selection' and trim !='Make Selection' and year !='Make Selection':
            if engine_size == 'No Selection':
                engine_size = -1

        # select engine size
        cylinders = st.selectbox("Cylinders",car_cylinders)

        if make !='Make Selection' and model !='Make Selection' and trim !='Make Selection' and year !='Make Selection':
            if cylinders == 'No Selection':
                cylinders = -1

        
        # select number of doors
        doors = st.selectbox("Doors",car_doors)

        if make !='Make Selection' and model !='Make Selection' and trim !='Make Selection' and year !='Make Selection':
            if doors == 'No Selection':
                doors = -1 



    with col3:

        st.header("Optional fields") 
        
        # select base exterior color
        base_exterior_color = st.selectbox("Exterior Color",exterior)

        # impute base exterior color from classifier
        if base_exterior_color == 'No Selection':
            if uploaded_file is not None:
                if message == 'Acceptable image':
                    base_exterior_color = color_msg

                else:
                    base_exterior_color = -1
  
        #select base interior color
        base_interior_color = st.selectbox("Interior Color",interior)

        if make !='Make Selection' and model !='Make Selection' and trim !='Make Selection' and year !='Make Selection':
            if base_interior_color == 'No Selection':
                base_interior_color = -1

        #select city mpg
        city_mpg = st.selectbox('City MPG',citympg)
        if make !='Make Selection' and model !='Make Selection' and trim !='Make Selection' and year !='Make Selection':
            if city_mpg == 'No Selection':
                city_mpg = -1

        #select highway mpg
        highway_mpg = st.selectbox('Highway MPG', highwaympg)
        if make !='Make Selection' and model !='Make Selection' and trim !='Make Selection' and year !='Make Selection':
            if highway_mpg =='No Selection':
                highway_mpg = -1



        #select state
        state = st.selectbox("State",states)

        if state == 'No Selection':
            state = -1

        # select is_certified
        is_certified = st.selectbox("Certified",("No","Yes"))

        if is_certified =="Yes":
            is_certified =1
        else:
            is_certified=0

        #select carfax_1_owner
        carfax_1_owner = st.selectbox("First owner",("No","Yes"))

        if carfax_1_owner == "Yes":
            carfax_1_owner=1
        else:
            carfax_1_owner=0

        #select carfax clean title
        carfax_clean_title = st.selectbox("Clean Title",("No","Yes"))

        if carfax_clean_title =="Yes":
            carfax_clean_title=1
        else:
            carfax_clean_title=0
    	




        # predict estimated vehicle value using given inputs from above.
        result = 0
        if st.button("Predict"):
            result=car_price_predictor(miles, year, make, model, trim, vehicle_type, body_type, drivetrain, fuel_type, engine_block, engine_size, transmission, doors, cylinders, 
        city_mpg, highway_mpg, base_exterior_color, base_interior_color, is_certified, state, carfax_1_owner, carfax_clean_title)
        if result !=0:
            st.success('Estimated value of your car is {} USD.  \nEstimated value lies in this range {} - {} USD.'.format(np.round(result[0]),np.round(result[1][0]),np.round(result[1][1])))



if __name__=='__main__':
    main()