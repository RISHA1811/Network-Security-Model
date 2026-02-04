import os,sys
from networksecurity.exception.exception import NetworkCustomException
from networksecurity.logging.logger import logging
from networksecurity.components.data_injection import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer

from networksecurity.entity.config_entity import(
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
)

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifacts,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact,
)

class TrainingPipeline:
    def __init__(self):
        self.trainingpipelineconfig=TrainingPipelineConfig()


    def start_data_ingestion(self):
        try:
            self.dataingestionconfig=DataIngestionConfig(self.trainingpipelineconfig)
            data_ingestion=DataIngestion(self.dataingestionconfig)
            logging.info("Hey lets go and initiate ")
            dataingestionartifact=data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Initialtion Completed{dataingestionartifact}")
            return dataingestionartifact
        except Exception as e:
            raise NetworkCustomException(e,sys)
    
    def start_data_validation(self,data_ingestion_artifact=DataIngestionArtifacts):
        try:
            data_validation_config=DataValidationConfig(training_pipeline_config=self.trainingpipelineconfig)
            data_validation=DataValidation(data_ingestion_artifact,data_validation_config)
            data_validation_artifact=data_validation.initiate__data_validation()
            logging.info(f"Data Validation {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise NetworkCustomException(e,sys)
        
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact):
        try:
            logging.info("Data Transformation Started")
            data_transformation_config=DataTransformationConfig(self.trainingpipelineconfig)
            data_transformation=DataTransformation(data_validation_artifact,data_transformation_config)
            data_transformation_artifact=data_transformation.initiate_data_transformation()
            logging.info(f"Data Transformation {data_transformation_artifact}")
            return data_transformation_artifact
        
        except Exception as e:
            raise NetworkCustomException(e,sys)

    def start_model_trainer(self,data_transformation_artifact:DataTransformationArtifact)->ModelTrainerArtifact:
        try:
            logging.info("Model Training Started")
            self.model_trainer_config=ModelTrainerConfig(self.trainingpipelineconfig)
            model_trainer=ModelTrainer(model_trainer_config=self.model_trainer_config,data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact=model_trainer.initiate_model_trainer()
            logging.info(f" Model Training Artifacts {model_trainer_artifact} ")
            return model_trainer_artifact

        except Exception as e:
            raise NetworkCustomException(e,sys)
    
    def run_pipeline(self):
        try:
            dataingestionartifact=self.start_data_ingestion(dataingestionartifact=dataingestionartifact)
            data_validation_artifact=self.start_data_validation(data_validation_artifact=data_validation_artifact)
            data_transformation_artifact=self.start_data_transformation(data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact=self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
            return model_trainer_artifact
        except Exception as e:
            raise NetworkCustomException(e,sys)
    
        







