import os
import sys 
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# parent_directory = os.path.abspath('.')
# print(parent_directory)
sys.path.append(os.path.abspath('.'))
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig





@dataclass
class DataIngestionConfig():
    raw_data_path: str =  os.path.join('artifacts', "data.csv")
    train_data_path: str =  os.path.join('artifacts', "train.csv")
    test_data_path: str =  os.path.join('artifacts', "test.csv")

class DataIngestion():
    def __init__(self):
        self.ingestionConfig = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info("Read the dataset as dataframe")

            os.makedirs(os.path.dirname(self.ingestionConfig.train_data_path), exist_ok=True)
            df.to_csv(self.ingestionConfig.raw_data_path, index=False, header=True)
            logging.info("Train & Test split initiated")

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=40)
            train_set.to_csv(self.ingestionConfig.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestionConfig.test_data_path, index=False, header=True)
            logging.info("Data ingetsion completed......")

            return (self.ingestionConfig.train_data_path, self.ingestionConfig.test_data_path)

        except Exception as e:
            # pass
            raise CustomException(e, sys)
        

if __name__ == '__main__':
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)