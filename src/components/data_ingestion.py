import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import mlflow


# Stores all file paths used during data ingestion.
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        # Create an object containing all file path configurations.
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        # Log the start of the data ingestion process.
        logging.info("Entered the data ingestion method or component")

        try:
            # Read the dataset into a Pandas DataFrame.
            df = pd.read_csv('/Users/dhruvitjalodhara/programming/ML Practice/Hotel Reservation Prediction/notebook/data/cleaned_data.csv')
            logging.info("Read the Dataset as Pandas Dataframe")

            # Create the artifacts directory if it doesn't exist.
            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path),
                exist_ok=True
            )

            # Save the original dataset.
            df.to_csv(
                self.ingestion_config.raw_data_path,
                index=False,
                header=True
            )

            logging.info("Train Test Split Initiated")

            test_size_ratio = 0.25
            random_state_seed = 42

            # Split the dataset into training and testing sets.
            train_set, test_set = train_test_split(
                df,
                test_size=test_size_ratio,
                random_state=random_state_seed
            )

            # Save the training dataset.
            train_set.to_csv(
                self.ingestion_config.train_data_path,
                index=False,
                header=True
            )

            # Save the testing dataset.
            test_set.to_csv(
                self.ingestion_config.test_data_path,
                index=False,
                header=True
            )

            logging.info("Ingestion of the data is completed")

            # 🚀 MLflow Integration Start
            if mlflow.active_run():
                logging.info("Logging data ingestion parameters to MLflow")
                
                # Log metadata parameters about data splitting and sizing
                mlflow.log_param("raw_data_rows", df.shape[0])
                mlflow.log_param("raw_data_cols", df.shape[1])
                mlflow.log_param("ingestion_test_size_ratio", test_size_ratio)
                mlflow.log_param("ingestion_random_state", random_state_seed)
                
                # Optional: Log the raw dataset as an artifact if it's small enough
                mlflow.log_artifact(self.ingestion_config.raw_data_path, artifact_path="raw_data")

            # Return paths for use in the next pipeline component.
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            # Raise a custom exception with detailed error information.
            raise CustomException(e, sys)
        


        