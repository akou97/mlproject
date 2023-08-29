import os, sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from sklearn.metrics import r2_score
import dill
def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as f:
            dill.dump(obj, f)
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train , X_test, y_test, models):
    try:
        report = {}
        for name_model, model in models.items():
            model.fit(X_train, y_train)
            y_train_pred  = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[name_model] = test_model_score
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)