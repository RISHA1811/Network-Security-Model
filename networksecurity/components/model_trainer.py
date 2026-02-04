import os
import sys
from networksecurity.exception.exception import NetworkCustomException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.utils.main_utils.utils import save_object,load_object, load_numppy_array_data
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import(
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow
from networksecurity.utils.main_utils.utils import evaluate_models
import dagshub
dagshub.init(repo_owner='rishabhjha1811', repo_name='Network-Security-Model', mlflow=True)



class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
        try:
            self.model_trainer_config=model_trainer_config
            self.data_transformation_artifact=data_transformation_artifact

        except Exception as e:
            raise NetworkCustomException(e,sys)
        

    def track_mlflow(self,best_model,classification_metric):
        with mlflow.start_run():
            fl_score=classification_metric.f1_score
            precision_score=classification_metric.precision_score
            recall_score=classification_metric.recall_score

            mlflow.log_metric("f1_score",fl_score)
            mlflow.log_metric("precision_score",precision_score)
            mlflow.log_metric("recall_score",recall_score)

            mlflow.sklearn.log_model(best_model,"model")


        
    def train_model(self,x_train,y_train, x_test,y_test):
        models={
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=100),
            "Adaboost": AdaBoostClassifier(),
        }
        params = {
    "Decision Tree": {
        'criterion': ['gini', 'entropy', 'log_loss'],
        # 'splitter': ['best', 'random'],
        # 'max_features': ['sqrt', 'log2'],
    },

    "Random Forest": {
        # 'criterion': ['gini', 'entropy', 'log_loss'],
        # 'max_features': ['sqrt', 'log2', None],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },

    "Gradient Boosting": {
        # 'loss': ['log_loss', 'exponential'],
        'learning_rate': [.1, .01, .05, .001],
        'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
        # 'criterion': ['squared_error', 'friedman_mse'],
        # 'max_features': ['auto', 'sqrt', 'log2'],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    },

    "Logistic Regression": {},

    "Adaboost": {
        'learning_rate': [.1, .01, 0.5, .001],
        'n_estimators': [8, 16, 32, 64, 128, 256]
    }
}
        
        models_report:dict=evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test, models=models, params=params)

        ## To get the best model score
        best_model_score=max(sorted(models_report.values()))

        ## to get best model name from the dict
        best_model_name = list(models_report.keys())[
            list(models_report.values()).index(best_model_score)
        ]
        best_model=models[best_model_name]
        y_train_pred=best_model.predict(x_train)
        classification_train_metric=get_classification_score(y_true=y_train,y_pred=y_train_pred)

        ## Track the mlflow
        ## To track the experiments and metrics values

        self.track_mlflow(best_model,classification_train_metric)








        y_test_pred=best_model.predict(x_test)
        classification_test_metric=get_classification_score(y_true=y_test,y_pred=y_test_pred)

        preprocessor=load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path=os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path,exist_ok=True)
        network_model = NetworkModel(
    preprocessor=preprocessor,
    model=best_model
)
        save_object(
    self.model_trainer_config.trained_model_file_path,
    obj=network_model
)
        

        save_object("final_model/model.pkl",best_model)
        


        ## Model Trainer Artifact
        modeltrainerartifact=ModelTrainerArtifact(trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                             train_metric_artifact=classification_train_metric,
                             test_metric_artifact=classification_test_metric
                             )
        logging.info(f"Model Trainer Artifacts{modeltrainerartifact}")
        return modeltrainerartifact
    




    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            train_file_path=self.data_transformation_artifact.transformed_train_file_path
            test_file_path=self.data_transformation_artifact.transformed_test_file_path
            ## Loading the training array and testing array
            train_arr=load_numppy_array_data(train_file_path)
            test_arr=load_numppy_array_data(test_file_path)

            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )
            model=self.train_model(x_train,y_train, x_test,y_test)

        except Exception as e:
            raise NetworkCustomException(e,sys)


