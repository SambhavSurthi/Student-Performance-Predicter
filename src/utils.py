import os
import sys
import dill
from src.logger import logging
from src.exception import CustomException

import pandas as pd
import numpy as np

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info(f"Object successfully saved at {file_path}")

    except Exception as e:
        logging.error("Error occurred while saving object")
        raise CustomException(e, sys)


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score


def evaluate_model(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}

        for model_name in models.keys():

            model = models[model_name]
            param_grid = param[model_name]

            # GridSearchCV
            gs = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=3,
                n_jobs=-1,
                scoring='r2'
            )

            gs.fit(X_train, y_train)

            # Get best model after tuning
            best_model = gs.best_estimator_

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            # Scores
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            print(model_name)
            print("Best CV Score:", gs.best_score_)
            print("Test Score:", test_score)

            report[model_name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
