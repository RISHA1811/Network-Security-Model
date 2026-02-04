import sys
import os
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from networksecurity.exception.exception import NetworkCustomException
from networksecurity.logging.logger import logging

from networksecurity.constant.training_pipeline import (
    TARGET_COLUMN,
    DATA_TRANSFORMATION_IMPUTER_PARAMS
)

from networksecurity.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact
)

from networksecurity.entity.config_entity import DataTransformationConfig
from networksecurity.utils.main_utils.utils import (
    save_numpy_array_data,
    save_object
)


class DataTransformation:

    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        data_transformation_config: DataTransformationConfig
    ):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise NetworkCustomException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkCustomException(e, sys)

    def get_data_transformer_object(self) -> Pipeline:
        logging.info("Entered get_data_transformer method of DataTransformation class")
        try:
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            logging.info("KNN Imputer initialized")

            processor = Pipeline(
                steps=[
                    ("imputer", imputer)
                ]
            )

            return processor

        except Exception as e:
            raise NetworkCustomException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("Entered initiate_data_transformation method")

        try:
            logging.info("Reading train and test data")
            train_df = self.read_data(
                self.data_validation_artifact.valid_train_file_path
            )
            test_df = self.read_data(
                self.data_validation_artifact.valid_test_file_path
            )

            # ===================== Train =====================
            input_feature_train_df = train_df.drop(
                columns=[TARGET_COLUMN], axis=1
            )
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)

            # ===================== Test ======================
            input_feature_test_df = test_df.drop(
                columns=[TARGET_COLUMN], axis=1
            )
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

            # 🔥 Drop non-numeric / ID columns
            input_feature_train_df = input_feature_train_df.drop(
                columns=["_id", "id"], errors="ignore"
            )
            input_feature_test_df = input_feature_test_df.drop(
                columns=["_id", "id"], errors="ignore"
            )

            # ================== Transformation ==================
            preprocessor = self.get_data_transformer_object()

            preprocessor_object = preprocessor.fit(input_feature_train_df)

            transformed_input_train = preprocessor_object.transform(
                input_feature_train_df
            )
            transformed_input_test = preprocessor_object.transform(
                input_feature_test_df
            )

            # ================== Final Arrays ==================
            train_arr = np.c_[
                transformed_input_train,
                np.array(target_feature_train_df)
            ]

            test_arr = np.c_[
                transformed_input_test,
                np.array(target_feature_test_df)
            ]

            # ================== Save Artifacts ==================
            save_numpy_array_data(
                self.data_transformation_config.transformed_train_file_path,
                train_arr
            )

            save_numpy_array_data(
                self.data_transformation_config.transformed_test_file_path,
                test_arr
            )

            save_object(
                self.data_transformation_config.transformed_object_file_path,
                preprocessor_object
                )
            
            save_object("final_model/preprocessor.pkl",preprocessor_object)








            data_transformation_artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logging.info("Data Transformation completed successfully")
            return data_transformation_artifact

        except Exception as e:
            raise NetworkCustomException(e, sys)
