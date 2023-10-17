import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import MaxAbsScaler




@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        '''
        This function is responsible for Data Transformation 
        '''
        try:
            numerical_columns = [
                'current_score',
                'balls_left',
                'wickets_left',
                'crr',
                'last_five'
            ]
            categorical_columns = [ 
                'batting_team', 
                'bowling_team', 
                'city'
            ]

            # num_pipeline= Pipeline(
            #     steps=[
            #     ("imputer",SimpleImputer(strategy="median")),
            #     ("scaler",StandardScaler())

            #     ]
            # )
            cat_pipeline=Pipeline(

                steps=[
                
                ("one_hot_encoder",OneHotEncoder(sparse=False,drop='first'))
                
                ]

            )
            
            logging.info(f'Categorical columns: {categorical_columns}')
            logging.info(f'Numerical columns: {numerical_columns}')

            preprocessor = ColumnTransformer(
                [
                    # ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ],
                remainder='passthrough'
            )

            return preprocessor
        

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformations(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)  
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data Completed!')
            logging.info('Obtaining Pre-processing Object')
        
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name =  "runs_x"
            numerical_columns = [
                'current_score',
                'balls_left',
                'wickets_left',
                'crr',
                'last_five'
                
            ]

            # input_feature_train_df = train_df.drop(columns = [target_column_name], axis = 1)
            # target_feature_train_df = train_df[target_column_name]
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns = [target_column_name], axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f'Applying preprocessing object on training dataset and training dataset.')

            # input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            # input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)


            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


            logging.info(f'Saved preprocessing object.')

            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )



        except Exception as e:
            raise CustomException(e, sys)
