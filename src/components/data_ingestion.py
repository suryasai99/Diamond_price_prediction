import os
import sys
from src.logger import logging
from src.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass 

## initialize the data indestion configuration

@dataclass
class DataIngestionconfig:
    train_data_path = os.path.join('artifacts','train.csv')
    test_data_path = os.path.join('artifacts','test.csv')
    raw_data_path = os.path.join('artifacts','raw.csv')

# create DataIngestion class:
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionconfig()
    
    def initiate_data_ingestion(self):
        logging.info('Data ingestion method starts')

        try:
            df = pd.read_csv(os.path.join('notebooks/data/gemstone.csv'))
            logging.info('original Dataset imported into dataframe from csv file')

            # In case If we dont have artifacts folder. we need to create it
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok = True)
            df.to_csv(self.ingestion_config.raw_data_path) # used if our data is not in csv format
            logging.info('Raw data is created')

            # dividing training and testing data
            train_data,test_data = train_test_split(df, test_size=0.3, random_state=42)

            # saving it in the artifacts path
            train_data.to_csv(self.ingestion_config.train_data_path, 
                              index = False,
                              header = True)
            test_data.to_csv(self.ingestion_config.test_data_path,
                             index = False,
                             header = True)
            
            logging.info('Ingestion of Data is completed')

            return(self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path)

        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e,sys)
