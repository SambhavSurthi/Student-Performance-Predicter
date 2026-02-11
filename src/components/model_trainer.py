import os
import sys
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass

# models

from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import LassoCV,RidgeCV,ElasticNetCV

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array, preprocessor_path):
        try:
            logging.info('Splitting training and test input data')

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'randomforest': RandomForestRegressor(),
                'decisiontree': DecisionTreeRegressor(),
                'gradientboost': GradientBoostingRegressor(),
                'catboost': CatBoostRegressor(verbose=False),
                'xgboost': XGBRegressor(),
                'linearregression': LinearRegression(),
                'adaboost': AdaBoostRegressor(),
                'knnregressor': KNeighborsRegressor(),
                'lasso': LassoCV(),
                'ridge': RidgeCV(),
                'elasticnet': ElasticNetCV()
            }

            model_report = evaluate_model(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models
            )

            best_model_score = max(model_report.values())

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No Best Model Found", sys)

            best_model.fit(X_train, y_train)

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"R2 Score: {r2}")

            return r2

        except Exception as e:
            raise CustomException(e, sys)
