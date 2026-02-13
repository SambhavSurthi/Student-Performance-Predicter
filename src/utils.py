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
        best_model_name = None
        best_model = None
        best_score = float("-inf")   # VERY IMPORTANT
        best_params = None


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

            print("Best Parameters:", gs.best_params_)

            report[model_name] = {
                "cv_score": gs.best_score_,
                "test_score": test_score,
                "best_params": gs.best_params_
            }

            # Track best model based on CV score
            if gs.best_score_ > best_score:
                best_score = gs.best_score_
                best_model_name = model_name
                best_model = best_model
                best_params = gs.best_params_

        print("\n✅ Best Model Selected:", best_model_name)
        print("Best CV Score:", best_score)
        print("Best Hyperparameters:", best_params)

        return  best_model_name, best_model, best_score, best_params, report

    except Exception as e:
        raise CustomException(e, sys)



def load_object(path):
    try:
        with open(path, 'rb') as fileObj:
            return dill.load(fileObj)
    except Exception as e:
        raise CustomException(e, sys)