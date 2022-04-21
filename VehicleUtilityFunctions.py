# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 15:27:39 2022

@author: rafeeshaik
"""

import pandas as pd
import numpy as np
import scipy.stats as st
import joblib

import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

X_cols=['miles', 'year', 'make', 'model', 'trim', 'vehicle_type', 'body_type',
       'drivetrain', 'fuel_type', 'engine_block', 'engine_size',
       'transmission', 'doors', 'cylinders', 'city_mpg', 'highway_mpg',
       'base_exterior_color', 'base_interior_color', 'is_certified', 'state',
       'carfax_1_owner', 'carfax_clean_title']


trim_lookup = pd.read_csv('trim_lookup.csv')
# use this method in the frontend (UI interface) to get the default trim if use doesn't provide the trim information
def getTrim(make, model):
    trim = ''
    try:
        trim= trim_lookup[(trim_lookup.make==make) & (trim_lookup.model==model)]['trim'].values[0]
    except:
        trim = model
    return trim


vin_lookup_year_df = pd.read_csv('vin_lookup_year.csv')
# Get the default vehicle specifications from based on Make, Model, Model Year and Trim.
def getVehicleDetails(make, model,trim=None, year=None):
    vehicleDetails={}
    vehicleDetails_col=['vehicle_type', 'body_type', 'drivetrain',
       'fuel_type', 'engine_block', 'engine_size', 'transmission', 'doors',
       'cylinders', 'city_mpg', 'highway_mpg','base_exterior_color','base_interior_color','state']
    
    if trim is None:
        trim=getTrim(make, model)
   
    if year is None:
        vehicleDetails=vin_lookup_year_df[(vin_lookup_year_df.make==make) &\
                             (vin_lookup_year_df.model==model) &\
                             (vin_lookup_year_df.trim==trim)].groupby(vehicleDetails_col).size().reset_index().sort_values(0, ascending=False).drop(0,axis=1).head(1).to_dict('r')[0]
        
    else:
        vehicleDetails=vin_lookup_year_df[(vin_lookup_year_df.year==year) &\
                           (vin_lookup_year_df.make==make) &\
                           (vin_lookup_year_df.model==model) &\
                           (vin_lookup_year_df.trim==trim)].groupby(vehicleDetails_col).size().reset_index().sort_values(0, ascending=False).drop(0,axis=1).head(1).to_dict('r')[0]
    if len(vehicleDetails)>0:
        return vehicleDetails
    else:
        return vin_lookup_year_df.groupby(vehicleDetails_col).size().reset_index().sort_values(0, ascending=False).drop(0,axis=1).head(1).to_dict('r')[0]


# This method will provide default vehicle specifications if user is not able to provide them.
# We have prepared a specifications lookup table from the 2.7 million vehiles we have from MarketWatch.
# This function works in a similar way to VIN lookup that industry uses.
def imputeX(single_predict):
    make = single_predict.get('make')
    model = single_predict.get('model')
    year = single_predict.get('year')
    trim = single_predict.get('trim')

    if trim is None:
        trim =getTrim(make, model)
        single_predict['trim']=trim
    
    available_cols = list(single_predict.keys())
    
    default_values = getVehicleDetails(make = make,model=model, trim=trim, year=year)
    
    missing_cols = [x for x in X_cols if x not in available_cols]

    for col in missing_cols:
        single_predict[col]=default_values.get(col,0)

    single_predict_df = pd.DataFrame(columns=single_predict.keys(), data = np.array(list(single_predict.values())).reshape(1, len(single_predict)))

    single_predict_df.miles= pd.to_numeric(single_predict_df.miles, errors='coerce')
    single_predict_df.year= pd.to_numeric(single_predict_df.year, errors='coerce')
    single_predict_df.engine_size= pd.to_numeric(single_predict_df.engine_size, errors='coerce')
    single_predict_df.doors= pd.to_numeric(single_predict_df.doors, errors='coerce')
    single_predict_df.cylinders = pd.to_numeric(single_predict_df.cylinders, errors='coerce')
    single_predict_df.city_mpg = pd.to_numeric(single_predict_df.city_mpg, errors='coerce')
    single_predict_df.highway_mpg = pd.to_numeric(single_predict_df.highway_mpg, errors='coerce')
    single_predict_df.is_certified = pd.to_numeric(single_predict_df.is_certified, errors='coerce')
    single_predict_df.carfax_1_owner = pd.to_numeric(single_predict_df.carfax_1_owner, errors='coerce')
    single_predict_df.carfax_clean_title= pd.to_numeric(single_predict_df.carfax_clean_title, errors='coerce')

    single_predict_df = single_predict_df[X_cols]

    return single_predict_df


# Load the RandomForest Model pipeline
rf_pipe = joblib.load('oe_rf_depth15_660k.sav')

def estimateVehileValue(X):
    
    estimated_value = rf_pipe.predict(X)
    x_treanformed=rf_pipe['preprocessor'].transform(X)
    rf_estimators = rf_pipe['estimator'].estimators_

    preds=[]
    for estimator in rf_estimators:
        preds.append(estimator.predict(x_treanformed))
        
      
    data = [x[0] for x in preds]
    
    fig, ax = plt.subplots(figsize=(10,5))
    sns.histplot(data=data, kde=True, bins=25)
    
    # 95%, 2 sigma confidence interval(CI)
    ci = st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data))
    return (estimated_value[0], ci)