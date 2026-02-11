import os
import sys

import pandas as pd
import numpy as np

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass  #used to create class variables

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig,ModelTrainer

@dataclass
class DataIngetionConfig:
    train_data_path:str=os.path.join('artifacts',"train.csv")
    test_data_path:str=os.path.join('artifacts',"test.csv")
    raw_data_path:str=os.path.join('artifacts',"raw.csv")


class DataIngetion:
    def __init__(self):
        self.ingetion_config=DataIngetionConfig()

    def initiate_data_ingetion(self): # used to read the database
        logging.info('Entered Data Ingetion method')

        try:
            logging.info('Reading The Dataset')
            dataset=pd.read_csv('notebook\data\student_data.csv')
            logging.info('Reading Successful')

            logging.info('Creating Folders to save the datasets')

            # create artifact folder
            os.makedirs(os.path.dirname(self.ingetion_config.train_data_path),exist_ok=True)
            dataset.to_csv(self.ingetion_config.raw_data_path,index=False,header=True)

            logging.info('Initiated train test split')

            train_set,test_set=train_test_split(dataset,test_size=0.3,random_state=42)

            train_set.to_csv(self.ingetion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingetion_config.test_data_path,index=False,header=True)

            logging.info('Train test Split Completed, Data Ingetion Successful.')

            return (
                self.ingetion_config.train_data_path,
                self.ingetion_config.test_data_path,
                self.ingetion_config.raw_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
        

if __name__=='__main__':
    print('File Running')
    obj=DataIngetion()
    train_path,test_path,_=obj.initiate_data_ingetion()

    data_transformation=DataTransformation()
    train_array,test_arr,preprocessor_path=data_transformation.initiated_data_transfomation(train_path,test_path)

    trainer=ModelTrainer()
    r2=trainer.initiate_model_training(train_array=train_array,test_array=test_arr,preprocessor_path=preprocessor_path)

    print(r2)


    print('Successful')
