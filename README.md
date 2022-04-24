# SIADS 697 Capstone Project
## Applying Machine Learning Models to Estimate Vehicle Value with Pictures and Partial Data

### Project Introduction
We embarked on this project with the goal of creating a version of online forms that leverages machine learning to be more friendly to partial information. Many online forms require all their fields to be filled in order to give a result. We wanted to give users the flexibility to enter only the information that they are comfortable with entering and still be able to provide them with value. Another goal of the project was to be transparent about the uncertainty associated with the result, as this was another aspect that seems to be lacking in online forms. 
Our project specifically is centered around the estimation of vehicle prices. We chose this focus because we all had some significant knowledge about vehicles, and it was still a complex enough subject that could lead to the online form having fields that people either didn't know or might be uncomfortable with sharing. For example, someone might be uncomfortable sharing their location which could affect vehicle values, or they might not know the size of the engine in their vehicle.
### Project Objective: 
To reach the goals we had set, we came up with a plan. We wanted to use whatever data the user was willing to give us about their vehicle to try to make a prediction of how much their vehicle would be worth. This data optionally included an image. The image in conjunction with any other data that the user was willing to give would be used for the prediction. To use the image, we would make image classifiers to get useful features about the car that would later be fed to our final model. The final model would then use all of the data given directly from the user and data extracted from the image to make a price prediction. Along with the model’s best guess, a range between two dollar amounts would be given. This range would be the range that the model thinks likely holds the true value of the car. This range would grow or shrink based on how much data was given to the model by the user. The map for how we would develop this system is shown below.
![image](https://user-images.githubusercontent.com/55704682/164984055-3bcd2440-6085-49d5-9f61-56403830f639.png)

## Reproducing the results
### Source Data for the application
The original source data file for the project is over 20GB, with over 2.7 million vehicles.Due to the limitation on file size we can upload on Github, we included a sample data file with 10000 vehicles. Data sample file name is 'cars_us_used_10k_sample.csv'
### Data Preparation
Please use the 'Capstone_EDA_v2.ipynb' jupyter notebook file to prepare the data for EDA and Machine learning models. This notebook has two sections, first section for data cleaning and imputations, second section to perorm EDA.
This notebook read the source data file and produces a cleaned version of data in CSV file format for EDA and Machine Learning algorithms.
### EDA
The second section of 'Capstone_EDA_v2.ipynb' reads the cleaned version of data produced in part-1 of this notebook, It produces several visualization those are very interesting, and some of the observations can be surprising.
![image](https://user-images.githubusercontent.com/55704682/164985052-ad6fdcba-adaf-43b1-84e0-879feb30a7e5.png)

### Machine Learing Models
The 'Capstone_OLS_RF_XGBoost.ipynb' reads the cleansed dataset produced by part-1 of 'Capstone_EDA_v2.ipynb' notebook. We have implemented OLS (Oridnary Leaset Squares), RandomForestRegressor and XGBoostRegressor models in this file. We have captured various performace metrics such as R2 score, MAE, MAPE and MSE from these models
