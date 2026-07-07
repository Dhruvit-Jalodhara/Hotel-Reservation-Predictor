import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import mlflow

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler , LabelEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# Stores the path where the preprocessor object will be saved.
@dataclass
class DataTransformationConfig:
     # Stores the path where the preprocessor object will be saved.
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

    # Stores the path where the label encoder will be saved.
    label_encoder_file_path = os.path.join('artifacts', 'label_encoder.pkl')


class DataTransformation:
    def __init__(self):

        # Create an object containing transformation configuration.
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):

        # Creates and returns the complete preprocessing pipeline.
        try:

            # Numerical features to be scaled.
            numerical_columns = [
                'no_of_adults', 
                'no_of_children', 
                'no_of_weekend_nights', 
                'no_of_week_nights', 
                'required_car_parking_space', 
                'lead_time','arrival_month', 
                'repeated_guest', 
                'no_of_previous_cancellations', 
                'avg_price_per_room', 
                'no_of_special_requests'
            ]

            # Categorical features to be encoded.
            categorical_columns = ['type_of_meal_plan', 'room_type_reserved', 'market_segment_type']

            # Pipeline for numerical features.
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            # Pipeline for categorical features.
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    # Ignore unseen categories during prediction.
                    ('one_hot_encoder', OneHotEncoder(drop='first', handle_unknown='ignore')),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info(f'Numerical Features : {numerical_columns}')
            logging.info(f'Categorical Features : {categorical_columns}')

            # Apply different pipelines to different column groups.
            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )

            logging.info('Numerical Columns Standard Scaling Completed')
            logging.info('Categorical Columns Encoding Completed')

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):

        try:
            # Load training and testing datasets.
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Reading Train and Test Data Completed")

            # Create the preprocessing pipeline.
            preprocessing_obj = self.get_data_transformer_object()
            
            # Create Label Encoder for target column.
            le = LabelEncoder()

            target_column = 'booking_status'

            # Separate input and target columns.
            input_feature_train_df = train_df.drop(columns=target_column)
            target_feature_train_df = train_df[target_column]

            input_feature_test_df = test_df.drop(columns=target_column)
            target_feature_test_df = test_df[target_column]

            logging.info('Applying preprocessing object on training and testing data')


            # Learn preprocessing parameters from training data.
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            
            # Apply the same preprocessing to testing data.
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)


            # Encode training target labels.
            target_feature_train_arr = le.fit_transform(target_feature_train_df)
            
            # Encode testing target labels using the same mapping.
            target_feature_test_arr = le.transform(target_feature_test_df)


            # Combine transformed features with the target column.
            train_arr = np.c_[
                input_feature_train_arr,
                target_feature_train_arr
            ]

            test_arr = np.c_[
                input_feature_test_arr,
                target_feature_test_arr
            ]

            logging.info('Saving preprocessing object.')

            # Save the fitted preprocessor for future predictions.
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Save the fitted label encoder for future predictions.
            save_object(
                file_path=self.data_transformation_config.label_encoder_file_path,
                obj=le
            )

            # MLflow Integration Start
            if mlflow.active_run():
                logging.info("Logging preprocessing artifacts and parameters to MLflow")
                
                # Log the saved .pkl files as artifacts in a dedicated folder
                mlflow.log_artifact(
                    self.data_transformation_config.preprocessor_obj_file_path, 
                    artifact_path="preprocessing_objects"
                )
                mlflow.log_artifact(
                    self.data_transformation_config.label_encoder_file_path, 
                    artifact_path="preprocessing_objects"
                )
                
                # Log the shape of the datasets to ensure data integrity is recorded
                mlflow.log_param("train_data_shape", f"{train_arr.shape}")
                mlflow.log_param("test_data_shape", f"{test_arr.shape}")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                self.data_transformation_config.label_encoder_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        