from networksecurity.exception.exception import NetworkCustomException
from networksecurity.logging.logger import logging

## Abhi dekh dataingestion mein maine artifacts waale files bana liye ha, aur abhi humko jo raw data ha aur jo train and test data ha usko usme store karna ha 

from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifacts

import os
import sys
from typing import List
from sklearn.model_selection import train_test_split
import pymongo
from pymongo.mongo_client import MongoClient
import pandas as pd
import numpy as np 


from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL=os.getenv('MONGO_DB_URL')

class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config=data_ingestion_config

        except Exception as e:
            raise NetworkCustomException(e,sys)
    
    def export_collection_as_dataframe(self):
        try:
            database_name=self.data_ingestion_config.database_name
            collection_name=self.data_ingestion_config.collection_name
            self.mongo_client=pymongo.MongoClient(MONGO_DB_URL)
            collection=self.mongo_client[database_name][collection_name]


            df=pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df.drop(['_id'], axis=1)
            df.replace({"na":np.nan}, inplace=True)
            return df


        except Exception as e:
            raise NetworkCustomException(e,sys)
        

    def export_data_into_feature_store(self,dataframe:pd.DataFrame):
        try:
            feature_store_file=self.data_ingestion_config.feature_store_file_path
            dir_path=os.path.dirname(feature_store_file)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file,index=False,header=True)
            return dataframe
        
        except Exception as e:
            raise NetworkCustomException(e,sys)
        


    def split_data_as_train_test_split(self,dataframe:pd.DataFrame):
        try:
            train_set,test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
            )
            logging.info("The test_train_split is done")

            dir_path=os.path.dirname(self.data_ingestion_config.training_file_path)

            os.makedirs(dir_path,exist_ok=True)
            train_set.to_csv(
                self.data_ingestion_config.training_file_path,index=False, header=True
            )

            test_set.to_csv(
                ## Yaha compulsary ha file ka pura path dena 
                self.data_ingestion_config.testing_file_path,index=False, header=True
            )

            logging.info("Data is get splitted and only model formation is remain")



        except Exception as e:
            raise NetworkCustomException(e,sys)

        
    def initiate_data_ingestion(self):
        try:
            dataframe=self.export_collection_as_dataframe()
            dataframe2=self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test_split(dataframe2)

            dataingestionartifacts=DataIngestionArtifacts(trained_file_path=self.data_ingestion_config.training_file_path,test_file_path=self.data_ingestion_config.testing_file_path)

            return dataingestionartifacts
        

        except Exception as e:
            raise NetworkCustomException(e,sys)