import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion

if __name__=='__main__':
    D_I = DataIngestion()
    train_path,test_path = D_I.initiate_data_ingestion()
    print(train_path, test_path)