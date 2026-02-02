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
DATA_VALIDATION_INVALID_DIR:str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_DIR:str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_file_name:str = "report.yaml"

