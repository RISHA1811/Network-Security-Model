# from flask import Falsk, request, render_templates
import sys
import os
from networksecurity.components.data_injection import DataIngestion
from networksecurity.exception.exception import NetworkCustomException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import DataIngestionConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig

if __name__=="__main__":
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig=DataIngestionConfig(trainingpipelineconfig)
        data_ingestion=DataIngestion(dataingestionconfig)
        logging.info("Hey lets go and initiate ")
        print(data_ingestion.initiate_data_ingestion())
    except Exception as e:
        raise NetworkCustomException(e,sys)





