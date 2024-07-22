import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts/preprocessor.pkl')
            model_path = os.path.join('artifacts/model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            predict = model.predict(data_scaled)

            return predict


        except Exception as e:
            logging.info('error occured at predict class in prediction pipeline')
            raise CustomException(e,sys)
        
class CustomData:
    def __init__(self,
                 carat,
                 table,
                 cut,
                 color,
                 clarity):
        self.carat = carat
        self.table = table
        self.cut = cut
        self.color = color
        self.clarity = clarity

    def get_data_as_dataframe(self):
        try:
            CustomData_input_dict = {
                'carat':[self.carat],
                'table':[self.table],
                'cut':[self.cut],
                'color':[self.color],
                'clarity':[self.clarity]
            }
            df = pd.DataFrame(CustomData_input_dict)
            logging.info('Dataframe gathered')

            return df

        except Exception as e:
            logging.info('error occured at CustomData class in prediction pipeline')
            raise CustomException(e,sys)


        
