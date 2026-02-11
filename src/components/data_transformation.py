
# In this data transformation we will be performing transformations like- Encding,standardization,normalization,handling missing values...etc transformations.

import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join('artifacts','preprocessor.pkl')



class DataTransformation:
    def __init__(self):
        self.data_tranformation_config=DataTransformationConfig()

    # Create Pickel files
    def get_data_transformer_obj(self):
        '''
        This function is responsible for Data Transformation

        '''
        try:
            
            numerical_cols=['reading_score', 'writing_score']
            categorical_cols=['gender', 'race_ethnicity', 'parental_level_of_education', 'lunch', 'test_preparation_course']

            logging.info('Numerical And Categorical features Extracted..')
            logging.info(numerical_cols)
            logging.info(categorical_cols)

            logging.info('Starting With Pipeline preperation')

            logging.info('Numerical Pipeline Initated')
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="median")), # we have some outlier to we use SimpleImputer to impute missing values 
                    ('Scaler',StandardScaler())
                ]
            )
            logging.info('Numerical Pipeline Successful')
            logging.info('categorical Pipeline initated')
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent')),
                    ('encoder',OneHotEncoder(drop='first',)),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )
            logging.info('Categorical Pipeline Successful')

            logging.info('Combining Numerical and Categorical Pipeline using Columntransformer')
            logging.info('Column Transformer Initiated')
            preprocessor=ColumnTransformer(
                transformers=[
                    ('num_pipeline',num_pipeline,numerical_cols),
                    ('col_pipeline',cat_pipeline,categorical_cols)
                ]
            )
            logging.info('Column transformer Successful')

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiated_data_transfomation(self,train_path,test_path):
        try:
            logging.info('Reading The datasets')
            train_dataset=pd.read_csv(train_path)
            test_dataset=pd.read_csv(test_path)

            logging.info('Dataset reading Successful')

            logging.info('Calling Preprocessor Object')
            preprocessor_obj=self.get_data_transformer_obj()
            target_feature='math_score'

            logging.info('Extrating input and target data for training and test dataset')
            input_feature_train_df=train_dataset.drop(columns=[target_feature],axis=1)
            target_feature_train_df=train_dataset[target_feature]

            input_feature_test_df=test_dataset.drop(columns=[target_feature],axis=1)
            target_feature_test_df=test_dataset[target_feature]

            logging.info('Extrating input and target data for training and test dataset Successful')
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)

            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            logging.info('Applying preprocessing object on training dataframe and testing dataframe.')

            logging.info('Implementating/saving Train,test Dataset')

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]

            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info('saving preprocessor obj into pkl file')

            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            logging.info('saving preprocessor obj into pkl file Successful')

            logging.info('preprocessing object on training dataframe and testing dataframe Successfull...')

            return (
                train_arr,test_arr,self.data_tranformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)

