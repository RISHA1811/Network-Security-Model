from networksecurity.entity.artifact_entity import classificationMetricArtifact
from networksecurity.exception.exception import NetworkCustomException
from sklearn.metrics import f1_score,precision_score,recall_score
import os,sys

def get_classification_score(y_true, y_pred) -> classificationMetricArtifact:
    try:
        model_f1_score = f1_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)

        classification_metric = classificationMetricArtifact(
            f1_score=model_f1_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score
        )

        return classification_metric
    except Exception as e:
        raise NetworkCustomException(e, sys) from e
