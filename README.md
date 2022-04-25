# SIADS 697 Capstone Project
## Applying Machine Learning Models to Estimate Vehicle Value with Pictures and Partial Data

### Project Introduction
Our project specifically is centered around the estimation of vehicle prices. We chose this focus because we all had some significant knowledge about vehicles, and it was still a complex enough subject that could lead to the online form having fields that people either didn't know or might be uncomfortable with sharing. For example, someone might be uncomfortable sharing their location which could affect vehicle values, or they might not know the size of the engine in their vehicle.  

We embarked on this project with the goal of creating a version of online forms that leverages machine learning to be more friendly to partial information. Many online forms require all their fields to be filled in order to give a result. We wanted to give users the flexibility to enter only the information that they are comfortable with entering and still be able to provide them with value. Another goal of the project was to be transparent about the uncertainty associated with the result, as this was another aspect that seems to be lacking in online forms. 

To reach the goals we had set, we came up with a plan. We wanted to use whatever data the user was willing to give us about their vehicle to try to make a prediction of how much their vehicle would be worth. This data optionally included an image. The image in conjunction with any other data that the user was willing to give would be used for the prediction. To use the image, we would make image classifiers to get useful features about the car that would later be fed to our final model. The final model would then use all of the data given directly from the user and data extracted from the image to make a price prediction. Along with the model’s best guess, a range between two dollar amounts would be given. This range would be the range that the model thinks likely holds the true value of the car. This range would grow or shrink based on how much data was given to the model by the user. The map for how we would develop this system is shown below.

Full project report is available on Medium: https://medium.com/@rshaik_57200/siads-697-capstone-project-d8e0842e6a16

Illustration of proposed pipeline that describes the workflow of our entire project
![image](https://user-images.githubusercontent.com/55704682/165000581-9f863646-f24e-464a-89d4-c13076d75fef.png)

## Reproducing Structured Data Preparation, EDA and Regression Models
Please download the entire folder as a zip file and uncompress the zip file on your local computer. This project uses Jupyter notebook environment for data preparation, EDA and Machine Learning models.
### Step-1: Source Data for the application
The original source data file for the project is over 20GB, with over 2.7 million vehicles.Due to the limitation on file size we can upload on Github, we included a sample data file with 10000 vehicles. Data sample file name is 'cars_us_used_10k_sample.csv'
### Step-2: Data Preparation, Run this step in Jupyter Notebook environment
Structured Data preparation was done in Jupyter Notebook, 'Capstone_EDA_v2.ipynb'.
Please use the 'Capstone_EDA_v2.ipynb' jupyter notebook file to prepare the data. This notebook has two sections, first section for data cleaning and imputations, second section to perorm EDA.
This notebook read the source data file and produces a cleansed version of data in CSV file format, that will be later used for EDA and Machine Learning algorithms.
### Step-3: EDA, Run this step in Jupyter Notebook environment
EDA was performed in the second section of 'Capstone_EDA_v2.ipynb' notebook. It reads the cleansed version of data produced in part-1 of this notebook, It produces several visualization those are very interesting, and some of the observations can be surprising.
### Step-4: Machine Learing Models, Run this step in Jupyter Notebook environment
Machine learning model are implemented in jupyter notebook file 'Capstone_OLS_RF_XGBoost.ipynb'.
The 'Capstone_OLS_RF_XGBoost.ipynb' reads the cleansed dataset produced by part-1 of 'Capstone_EDA_v2.ipynb' notebook. We have implemented OLS (Oridnary Leaset Squares), RandomForestRegressor and XGBoostRegressor models in this file. We have captured various performace metrics such as R2 score, MAE, MAPE and MSE from these models. The trained pipelines are saved in pickle files. The final production deployed pickel file is 'oe_rf_depth20_50Estimators.sav' 
### Step-5: Import the custom utility methods we created for the application, save it in the same folder where the other Jupyter Notebooks are.
We have created a python script with all the custom utility methods used in the project. The script name is 'VehicleUtilityFunctions.py'. Please save it to the same folder where you have saved other jupyter notebok files. 
### Step-5: Real world test of ML model: Run this step in Jupyter Notebook environemnt.
We have created a jupyter notebook .SinglePrediction_RF.ipynb' to test the deployed version of ML model.
In this notebook one can enter the vehile details in a dictionary format and run the notebook to get the price prediction and 95% Confidence Interval range.

## Recreating Image Classifiers
### Step-1:
To recreate the CNN models first download images from the csv (cars_us_used_10k_sample.csv) using the “downloader.ipynb”. 
### Step-2:
Run the “filter_jpegs.py” on the image directory made from “downloader.ipynb”. This will make sure that all that is left in the directory is jpeg files.
### Step-3:
Use the saved model (packaged_exterior_classifier) to filter only acceptable images. This will get rid of all the images that are useless to the rest of the process. 
### Step-4:
Use the “create_CNN_dataset.ipynb” with the acceptable images as input to create the datasets for the CNN models. After that, the models can be trained based on their respective dataset. 
### Step-5:
Run the “color_classifier.py”,” body_classifier.py”, and “make_classifier.py” scripts with their respective datasets to train each model. 
### Step-6:
The “color_model_decoder.py”, “body_model_decoder.py”, and “make_model_decoder.py” can be used to test the model on individual images. Alternatively, we provided saved model that you can test they are: “packaged_color_classifier”, “packaged_body_classifier”, and “packaged_make_classifier”. 

## Creating Pipeline and Making Deployment

### Step-1:
The following command will install the packages according to the configuration file requirements.txt.
'pip install -r requirements.txt'

Put requirements.txt in the directory where the command will be executed. If it is in another directory, specify its path like path/to/requirements.txt.

### Step-2:
The following command will install the streamlit
'pip install streamlit'

### Step-3:
The following command will run the web interface
'streamlit run app.py'
