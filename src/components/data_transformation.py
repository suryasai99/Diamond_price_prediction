import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
import sys,os
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts/preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def Transformation_pipeline(self,df):
        try:
            logging.info('Dividing cols in data into numerical and categorical')
            # categorical and numerical cols
            numerical_cols = df.columns[df.dtypes!='object']
            categorical_cols = df.columns[df.dtypes=='object']

            # Define the custom ranking for each ordinal variable
            cut_categories = [
                'Fair',
                'Good',
                'Very Good',
                'Premium',
                'Ideal'
            ]
            color_categories = [
                'D','E',
                'F','G',
                'H','I','J'
            ]    
            clarity_categories = [
                'I1','SI2','SI1',
                'VS2','VS1','VVS2', 
                'VVS1','IF' 
            ]

            logging.info('Data transformation pipeline initiated')

            # creating a pipeline
            num_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy = 'median')),
                    ('scaler',StandardScaler())
                ]
                
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer',SimpleImputer(strategy = 'most_frequent')),
                    ('scaler',StandardScaler()),
                    ('ordinalencoder',OrdinalEncoder(categories = [cut_categories, color_categories, clarity_categories]))
                ]
            )

            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])
            

            logging.info('Data transformation completed')

            return preprocessor

        except Exception as e:
            logging.info('exception occured during data transformation pipeline')
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            logging.info('Data Transformation initiated')
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            #logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            #logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            independent_train = train_df.drop(['price','x','y','z','depth','id'],axis = 1)
            independent_test = test_df.drop(['price','x','y','z','depth','id'],axis = 1)
            target_train = train_df['price']
            target_test = test_df['price']

            logging.info(f'Train Dataframe Head : \n{independent_train.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{independent_test .head().to_string()}')

            logging.info('Obtaining preprocessing object')
            data_trans_pipeline = self.Transformation_pipeline(independent_train)

            ## apply the transformation pipeline
            independent_train_arr = data_trans_pipeline.fit_transform(independent_train)
            independent_test_arr = data_trans_pipeline.transform(independent_test)

            logging.info("Applied Transformation pipeline on training and testing datasets.")

            train_arr = np.c_(independent_train_arr, np.array(target_train))
            test_arr = np.c_(independent_test_arr, np.array(target_test))

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = data_trans_pipeline
            )

            logging.info('Transformation pipeline pickle file created and saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info('Error in Data Transformation')
            raise CustomException(e,sys)
