# importing libraries
import pandas as pd
import numpy as np
import sys,os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.logger import logging
from src.exception import CustomException
from src.utils import(save_object,
                      save_numpy_array_data)
from src.entity.artifact_entity import(DataIngestionArtifact,
                                       DataTransformationArtifact)
from src.entity.config_entity import DataTransformationConfig
from src.components.data_ingestion import DataIngestion
from src.constants.training_pipeline import *

class DataTransformation:
    def __init__(self,
                 data_ingestion_artifact:DataIngestionArtifact,
                 data_transformation_config:DataTransformationConfig):
        
        self.data_ingestion_artifact = data_ingestion_artifact
        self.data_transformation_config = data_transformation_config


    def get_data_transformation_object(self,data):        
         try:
            logging.info('Data Transformation initiated')
            # Define which columns should be ordinal-encoded and which should be scaled

            categorical_cols = [col for col in data.columns if data[col].dtype == 'O' ]
            numerical_cols = [col for col in data.columns if data[col].dtype != 'O' ]
            
            # Define the custom ranking for each ordinal variable
            cut_categories = ['Fair', 'Good', 'Very Good','Premium','Ideal']
            color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']
            
            logging.info('Pipeline Initiated')

            ## Numerical Pipeline
            num_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())

                ]
            )

            # Categorigal Pipeline
            cat_pipeline=Pipeline(
                steps=[
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                ('scaler',StandardScaler())
                ]
            )

            # merging numerical and categorical pipelines with column transformer
            preprocessor=ColumnTransformer([
            ('num_pipeline',num_pipeline,numerical_cols),
            ('cat_pipeline',cat_pipeline,categorical_cols)
            ])

            logging.info('Pipeline Completed')
            return preprocessor

         except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e,sys)
         

    def initiate_data_transformation(self):
        try:
            # Reading train and test data
            train_df = DataIngestion.read_data(self.data_ingestion_artifact.train_filepath)
            test_df = DataIngestion.read_data(self.data_ingestion_artifact.test_filepath)

            logging.info('imported train and test data')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            ## independent and dependent features
            x_train_df = train_df.drop(TARGET,axis=1)
            y_train_df=train_df[TARGET]
            x_test_df=test_df.drop(TARGET,axis=1)
            y_test_df=test_df[TARGET]
            logging.info('seperated independent and dependent features')

            # preprocessing object
            preprocessing_obj = self.get_data_transformation_object(x_train_df)

            ## apply the transformation for x_train and x_test
            input_feature_train_arr = preprocessing_obj.fit_transform(x_train_df)
            logging.info(f'preprocessing of x_train completed {input_feature_train_arr[1:5,:]}')
            input_feature_test_arr = preprocessing_obj.transform(x_test_df)
            logging.info(f'preprocessing of x_test completed {input_feature_test_arr[1:5,:]}')
            
            # saving numpy array of x_train
            save_numpy_array_data(
                file_path = self.data_transformation_config.x_train_filepath,
                array = input_feature_train_arr
            )

            # saving numpy array of x_test
            save_numpy_array_data(
                file_path = self.data_transformation_config.x_test_filepath,
                array = input_feature_test_arr
            )

            # saving numpy array of y_train
            save_numpy_array_data(
                file_path = self.data_transformation_config.y_train_filepath,
                array = y_train_df
            )

            # saving numpy array of y_test
            save_numpy_array_data(
                file_path = self.data_transformation_config.y_test_filepath,
                array = y_test_df
            )
            logging.info('saved all the numpy array files')

            # saving preprocessor pickle file
            save_object(
                file_path = self.data_transformation_config.preprocessor_filepath,
                obj = preprocessing_obj
            )
            logging.info('saved preprocessor pickle file')

            # saving the artifacts
            data_transformation_artifacts = DataTransformationArtifact(
                x_train_filepath = self.data_transformation_config.x_train_filepath,
                y_train_filepath = self.data_transformation_config.y_train_filepath,
                x_test_filepath = self.data_transformation_config.x_test_filepath,
                y_test_filepath = self.data_transformation_config.y_test_filepath,
                preprocessor_filepath = self.data_transformation_config.preprocessor_filepath
            )

            logging.info('data transformation artifacts created')
            return data_transformation_artifacts
        
        except Exception as e:
            logging.info('error occured in inititate_data_transformation module')
            raise CustomException(e,sys)