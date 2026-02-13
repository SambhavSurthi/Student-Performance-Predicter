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
class DataIngestionConfig:
    '''
    Docstring for DataIngestionConfig
    # These are the input paths which we need to save out files
    # in config we save all of our inputs needed. 
    # specifically we use os.join because joining path is different for every system example 
        windows: src/components
        linux: src\components
    as we are implementating modular programming our project must run in evey system. so we apply this method to solve
    '''
    
    
    train_data_path:str=os.path.join('artifacts',"train.csv") 
    test_data_path:str=os.path.join('artifacts',"test.csv")
    raw_data_path:str=os.path.join('artifacts',"raw.csv")


class DataIngestion:
    '''
    Docstring for DataIngestion
    
    # In this Class we perform DataIngestion Steps. 
    Example:
        # Loading a Dataset from a cloud database like MongoDB, so reading from out local device. these are steps are performed in this class
        # this class is also responsible for initiating train test split and saving the train and test data.
        
    All of these are performed in this section
    '''
    
    
    def __init__(self):
        '''
        Docstring for __init__
        
        :param self: Description
        # In this we will create Object for our DataIngestionConfig Class So that the paths/methods Initiated in the DataIngestionConfig Class Can Be used here
        
        now we can access the path as self.ingestion_config.raw_data_path to access the path
        '''
        
        
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self): # used to read the database
        '''
        Docstring for initiate_data_ingestion
        
        :param self: Description
        # This method is used to Load/read the Dataset, this can be your cloud Database or Local Database. the code to read the dataset is written here.
        '''
        logging.info('Project Starting...')
        
        logging.info('Entered Data Ingestion method')

        try:
            logging.info('Reading The Dataset')
            dataset=pd.read_csv('notebook\data\student_data.csv')
            logging.info('Reading Successful')
            

            logging.info('Creating Folders to save the datasets')
            # create artifact folder
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            logging.info('Folder Creation Successful')
            
            
            logging.info('Saving The Raw Dataset')
            dataset.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('Saving The Raw Dataset Successful')


            logging.info('Initiated train test split')
            train_set,test_set=train_test_split(dataset,test_size=0.3,random_state=42)
            logging.info('Initiated train test split Successful')
            
            
            logging.info('Saving The Train Dataset')
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            logging.info('Saving Train Dataset Successful')
            
            logging.info('Saving The Test Dataset')
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Saving The Test Dataset Successful')

            logging.info('Train test Split Completed, Data Ingestion Successful.')

            return (
                #    To Use the Paths in Future we return them.
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.raw_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)



if __name__=='__main__':
    print('File Running')
    obj=DataIngestion()
    train_path,test_path,_=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_array,test_arr,preprocessor_path=data_transformation.initiated_data_transfomation(train_path,test_path)

    trainer=ModelTrainer()
    r2=trainer.initiate_model_training(train_array=train_array,test_array=test_arr,preprocessor_path=preprocessor_path)

    print(r2)


    print('Successful')