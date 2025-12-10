import os
import sys
import pandas as pd
import numpy as np
from src.exeption import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        pd.to_pickle(obj, file_path)
    except Exception as e:
        raise CustomException(f"Error saving object: {e}", sys)

def evaluate_model(X_train, y_train, X_test, y_test, models, params):
    try:
        report={}
        for i in range(len(list(models))):
            model=list(models.values())[i]
            model_name=list(models.keys())[i]
            para=params[list(models.keys())[i]]
            gs=GridSearchCV(model, para, cv=3, n_jobs=-1, verbose=1)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            
            model.fit(X_train, y_train)
            
            y_train_pred=model.predict(X_train)

            
            y_test_pred=model.predict(X_test)
            
            train_model_score=r2_score(y_train, y_train_pred)
            
            test_model_score=r2_score(y_test, y_test_pred)
            
            report[model_name]=test_model_score

        return report
    except Exception as e:
        raise CustomException(e,sys)