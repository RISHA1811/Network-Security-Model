import os , sys
import numpy as np
import pandas as pd


## COMMON CONSTANT VARIABLES FOR TRAINING PIPELINES

TARGET_COLUMN="Result"
PIPELINE_NAME:str = "NetworkSecurity"
ARTIFACT_DIR:str = 'Artifacts'
FILE_NAME:str = 'NetworkData.csv'
TRAIN_FILE_NAME:str = "train.csv"
TEST_FILE_NAME:str = 'test.csv'


SCHEMA_FILE_PATH = os.path.join("data_schema","schema.yaml")

SAVED_MODEL_DIR=os.path.join("saved_models")
MODEL_FILE_NAME="model.pkl"



## Yaha hamare saare kaam ke cheeze rahenge , database ka naame , and etc sab

DATA_INGESTION_COLLECTION_NAME:str = 'NetowrkData'
DATA_INGESTION_DATABASE_NAME:str = 'RJ'
DATA_INGESTION_DIR_NAME:str = 'data_ingestion'
DATA_INGESTION_FEATURE_STORE_DIR:str = 'feature_store'
DATA_INGESTION_INGESTED_DIR:str = 'ingested'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION:float = 0.2

"""
Constants for the Data_Validation 
"""
DATA_VALIDATION_DIR_NAME:str = "data_validation"
DATA_VALIDATION_VALID_DIR:str = "validated"
DATA_VALIDATION_INVALID_DIR:str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR:str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_file_name:str = "report.yaml"


"""
Data Transformation Related Constant Start With Data_Transformation 
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR:str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR:str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME:str="preprocessing.pkl"


## KNN VALUE IS TO REPLACE THE NAN-VALUES
DATA_TRANSFORMATION_IMPUTER_PARAMS:dict = {
    "missing_values":np.nan,
    "n_neighbors":3,
    "weights":"uniform",

}

DATA_TRANSFORMATION_TRAIN_FILE_PATH:str = "train.npy"
DATA_TRANSFORMATION_TEST_FILE_PATH:str = "test.npy"



## Constant Of  Model Training

MODEL_TRAINER_DIR_NAME:str="model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR:str="trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME:str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE:float=0.6
MODEL_TRAINER_OVER_FITTING_UNDER_FITTING_THRESHOLD:float = 0.05



