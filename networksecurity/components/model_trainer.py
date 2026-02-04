import os
import sys
from networksecurity.exception.exception import NetworkCustomException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numppy_array_data,
    evaluate_models
)
from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)

import mlflow


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact
    ):
        try:
            self.model_train
            er_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkCustomException(e, sys)

    def track_mlflow(self, best_model, classification_metric):
        with mlflow.start_run():
            mlflow.log_metric("f1_score", classification_metric.f1_score)
            mlflow.log_metric("precision_score", classification_metric.precision_score)
            mlflow.log_metric("recall_score", classification_metric.recall_score)
            mlflow.sklearn.log_model(best_model, "model")

    def train_model(self, x_train, y_train, x_test, y_test):
        models = {
            "Random Forest": RandomForestClassifier(verbose=1),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(verbose=1),
            "Logistic Regression": LogisticRegression(max_iter=200),
            "Adaboost": AdaBoostClassifier(),
        }

        params = {
            "Decision Tree": {
                "criterion": ["gini", "entropy", "log_loss"],
            },
            "Random Forest": {
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.01, 0.05, 0.001],
                "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
            "Logistic Regression": {},
            "Adaboost": {
                "learning_rate": [0.1, 0.01, 0.5, 0.001],
                "n_estimators": [8, 16, 32, 64, 128, 256],
            },
        }

        models_report = evaluate_models(
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            models=models,
            params=params,
        )

        best_model_score = max(models_report.values())
        best_model_name = list(models_report.keys())[
            list(models_report.values()).index(best_model_score)
        ]

        best_model = models[best_model_name]

        y_train_pred = best_model.predict(x_train)
        classification_train_metric = get_classification_score(
            y_true=y_train,
            y_pred=y_train_pred,
        )

        self.track_mlflow(best_model, classification_train_metric)

        y_test_pred = best_model.predict(x_test)
        classification_test_metric = get_classification_score(
            y_true=y_test,
            y_pred=y_test_pred,
        )

        preprocessor = load_object(
            file_path=self.data_transformation_artifact.transformed_object_file_path
        )

        model_dir_path = os.path.dirname(
            self.model_trainer_config.trained_model_file_path
        )
        os.makedirs(model_dir_path, exist_ok=True)

        # ✅ FIXED: do NOT overwrite the class name
        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=best_model,
        )

        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=network_model,
        )

        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric,
        )

        logging.info(f"Model Trainer Artifacts: {model_trainer_artifact}")
        return model_trainer_artifact

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numppy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numppy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1],
            )

            return self.train_model(x_train, y_train, x_test, y_test)

        except Exception as e:
            raise NetworkCustomException(e, sys)
