import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_model
import sys,os
from dataclasses import dataclass
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor

@dataclass
class ModelTrainerconfig:
    trained_model_file_path = os.path.join('artifacts/model.pkl')

class ModelTrainer:
    def __init__(self):
        self.trained_model_config = ModelTrainerconfig()

    def initiate_model_training(self,train_array,test_array):
        try:
            logging.info('splitting dependent and independent variables from train and test data')
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                'linear_regression':LinearRegression(),
                'lasso':Lasso(),
                'ridge':Ridge(),
                'elastic_net':ElasticNet(),
                'random_forest': RandomForestRegressor(),
                'DT_regressor': DecisionTreeRegressor(),
                'KNN_regressor':KNeighborsRegressor()
            }

            model_report = evaluate_model(x_train,y_train,x_test,y_test,models)
            logging.info(f'Model Report : {model_report}')

            # TO get best model from dictionary
            best_model_score = max(sorted(model_report.values()))

            # To get best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            print(f'best model name: {best_model_name} , r2score : {best_model_score}')
            logging.info(f'best model name: {best_model_name} , r2score : {best_model_score}')

            save_object(
                file_path = self.trained_model_config.trained_model_file_path,
                obj = best_model

            )


        except Exception as e:
            logging.info('error occured in training')
            raise CustomException(e,sys)