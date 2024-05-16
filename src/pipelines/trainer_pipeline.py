import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__=='__main__':
    D_I = DataIngestion()
    train_path,test_path = D_I.initiate_data_ingestion()
    print(train_path, test_path)
    data_transformation=DataTransformation()

    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_path,test_path)

    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr,test_arr)