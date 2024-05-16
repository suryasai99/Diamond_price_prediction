import pandas as pd
import numpy as np
import sys,os,pickle
from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok = True) 

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(x_train,y_train,x_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            # Train model
            model.fit(x_train, y_train)

            # predict testing data
            y_pred = model.predict(x_test)

            # get r2 scores for train and test data 
            test_model_score = r2_score(y_test,y_pred)

            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        logging.info('Exception occured in the load_object function in utils.py')
        raise CustomException(e,sys)
