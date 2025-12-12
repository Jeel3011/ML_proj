import os
import sys
from src.exeption import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from src.utils import save_object,evaluate_model
from sklearn.metrics import  r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str=os.path.join('artifacts', 'model.pkl')

class ModelTrainer: 
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model Trainer initiated")
            logging.info("Splitting training and testing input data")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(),
                "Lasso": Lasso(),
                "ElasticNet": ElasticNet(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()
            }


            params = {
                "Linear Regression": {},
                "Ridge": {"alpha": [0.1,1.0,10.0]},
                "Lasso": {"alpha": [0.1,1.0,10.0]},
                "ElasticNet": {"alpha": [0.1,1.0,10.0], "l1_ratio": [0.1,0.5,0.9]},
                "K-Neighbors Regressor": {"n_neighbors": [3,5,7]},
                "Decision Tree": {"max_depth": [3,5,7]},
                "Random Forest": {"n_estimators": [50,100,200], "max_depth": [3,5,7]},
                "AdaBoost Regressor": {"n_estimators": [50,100,200]}
            }
            # best model score from dict
            model_report:dict=evaluate_model(X_train=x_train, y_train=y_train, X_test=x_test, y_test=y_test, models=models,params=params)

            #best model name from dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]


            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info(f"Best model found: {best_model_name} with r2 score: {best_model_score}")


            #save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(x_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
        

        except Exception as e:
            raise CustomException(e, sys)



