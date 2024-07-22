import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
import pandas as pd
from src.entity.config_entity import(ModelEvaluationConfig,
                                     ModelTrainingConfig,
                                     DataTransformationConfig,
                                     TrainingPipelineConfig)


class PredictPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
        self.model_training_config = ModelTrainingConfig(self.training_pipeline_config)
        self.model_evaluation_config = ModelEvaluationConfig(self.training_pipeline_config)

    def predict(self,features):
        try:
            preprocessor_path = self.data_transformation_config.preprocessor_filepath
            model_path = self.model_training_config.model_training_file_path

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            predict = model.predict(data_scaled)

            return predict


        except Exception as e:
            logging.info('error occured at predict class in prediction pipeline')
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self, carat,
                 table, cut, 
                 color,clarity,
                 depth, x,
                 y, z):
        self.carat = carat
        self.table = table
        self.cut = cut
        self.color = color
        self.clarity = clarity
        self.depth = depth
        self.x = x
        self.y = y
        self.z = z

    def get_data_as_dataframe(self):
        try:
            CustomData_input_dict = {
                'carat' : [self.carat],
                'table' : [self.table],
                'cut' : [self.cut],
                'color' : [self.color],
                'clarity' : [self.clarity],
                'depth' : [self.depth],
                'x' : [self.x],
                'y' : [self.y],
                'z' : [self.z],
            }
            
            df = pd.DataFrame(CustomData_input_dict)
            logging.info('Dataframe gathered')

            return df

        except Exception as e:
            logging.info('error occured at CustomData class in prediction pipeline')
            raise CustomException(e,sys)


        
