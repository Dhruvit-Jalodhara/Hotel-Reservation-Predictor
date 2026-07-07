import os
import sys
import mlflow
from datetime import datetime
import dagshub

from src.logger import logging
from src.exception import CustomException

# Import all the components
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


class TrainPipeline:
    def __init__(self):
        pass

    def run_pipeline(self):
        try:

            # 🔗 STEP 1: Initialize DagsHub Integration
            dagshub.init(repo_owner='Dhruvit-Jalodhara', repo_name='Hotel-Reservation-Predictor', mlflow=True)

            # 🧪 STEP 2: Set MLflow experiment name
            mlflow.set_experiment("Hotel_Reservation_Prediction")

            # 🕒 Create a unique run name using the current time
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            unique_run_name = f"Training_Pipeline_{timestamp}"

            # Set MLflow tracking server. , for local server
            # mlflow.set_tracking_uri("http://127.0.0.1:8000")


            # Start the parent MLflow run.
            with mlflow.start_run(run_name=unique_run_name) as active_run:

                logging.info(f"MLflow Parent Run Started. Run ID: {active_run.info.run_id}")
                print(f"Starting Training Pipeline... (MLflow Run ID: {active_run.info.run_id})")
                print(f"Run Name: {unique_run_name}")

                # 1️⃣ Data Ingestion Phase
                logging.info("--- Phase 1: Data Ingestion ---")
                print("Phase 1: Running Data Ingestion...")

                # Create Data Ingestion object.
                data_ingestion = DataIngestion()

                # Read and split the dataset.
                train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()

                # 2️⃣ Data Transformation Phase
                logging.info("--- Phase 2: Data Transformation ---")
                print("Phase 2: Running Data Transformation...")

                # Create Data Transformation object.
                data_transformation = DataTransformation()

                # Preprocess the training and testing data.
                train_arr, test_arr, preprocessor_path, le_path = (
                    data_transformation.initiate_data_transformation(
                        train_data_path,
                        test_data_path
                    )
                )

                # 3️⃣ Model Training and Evaluation Phase
                logging.info("--- Phase 3: Model Training ---")
                print("Phase 3: Running Model Training and Tuning... (This might take a moment)")

                # Create Model Trainer object.
                model_trainer = ModelTrainer()

                # Train the model and get the best F1 score.
                best_model_name, best_model_f1_score = model_trainer.initiate_model_training(
                    train_arr,
                    test_arr
                )

                # Pipeline Completion
                print("\nPipeline Completed Successfully!")

                print(f"Best Model Found: {best_model_name}") # 🛠️ Prints the best model name!
                print(f"Champion Model Test F1 Score: {best_model_f1_score:.4f}")

                #print("Go to http://127.0.0.1:8000 in your browser to see the results!")
                print("Go to https://dagshub.com/Dhruvit-Jalodhara/Hotel-Reservation-Predictor/experiments in your browser to see the results!")

                logging.info(f"Pipeline Completed. Champion: {best_model_name} | F1 Score: {best_model_f1_score}")

                logging.info(f"Pipeline Completed. Champion F1 Score: {best_model_f1_score}")

        except Exception as e:
            logging.error("Pipeline failed during execution.")
            print("\nPipeline Failed! Check the logs for details.")
            raise CustomException(e, sys)
        
    



if __name__ == "__main__":
    # Create and run the training pipeline.
    pipeline = TrainPipeline()
    pipeline.run_pipeline()


# Run this command before executing the pipeline: for local server
# mlflow server --host 127.0.0.1 --port 8000