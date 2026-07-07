import os
import sys
import time
import numpy as np
import pandas as pd
import dill
import mlflow
from sklearn.metrics import f1_score
from sklearn.model_selection import RandomizedSearchCV
from src.hyperparameters import MODEL_PARAMS

from src.exception import CustomException



# Saves any Python object (model, pipeline, encoder, etc.) to a file.
def save_object(file_path, obj):
    try:
        # Get the directory where the object will be saved.
        dir_path = os.path.dirname(file_path)

        # Create the directory if it doesn't already exist.
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in binary write mode and save the object.
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # Raise a custom exception with detailed debugging information.
        raise CustomException(e, sys)
    

# Train and evaluate multiple machine learning models.
def evaluate_models(x_train, y_train, x_test, y_test, models):
    try:
        # List to store each models performance.
        model_list = []
        train_f1_score = []
        test_f1_score = []

        # Dictionary to store trained models.
        trained_models = {}

        # Iterate through all models.
        for name, model in models.items():
                
            # Start a nested run for each individual model
            with mlflow.start_run(run_name=name, nested=True):
    
                # Get hyperparameters for current model.
                params = MODEL_PARAMS.get(name, {})

                # Perform hyperparameter tuning if parameters exist.
                if params:
                    model, best_params, best_cv_score = tune_model(
                        model=model,
                        params=params,
                        x_train=x_train,
                        y_train=y_train
                    )
                else:
                    model.fit(x_train, y_train)

                # Save trained model.
                trained_models[name] = model

                # Make predictions on training and testing data.
                y_train_pred = model.predict(x_train)
                y_test_pred = model.predict(x_test)

                # Calculate F1 score for both datasets.
                train_model_score = f1_score(y_train, y_train_pred)
                test_model_score = f1_score(y_test, y_test_pred)
                
                # Calculate train-test gap.
                gap = abs(train_model_score - test_model_score)

                # Log metrics to MLflow
                mlflow.log_metric("train_f1", train_model_score)
                mlflow.log_metric("test_f1", test_model_score)
                mlflow.log_metric("f1_score_gap", gap)
                
                mlflow.sklearn.log_model(
                    sk_model=model, 
                    name=name.replace(" ", "_"), 
                    serialization_format="cloudpickle"
                )

                # Store the testing score in the report.
                model_list.append(name)
                train_f1_score.append(train_model_score)
                test_f1_score.append(test_model_score)

        # Build performance summary
        score_df = pd.DataFrame(
            {
                'Model Name' : model_list,
                'Train f1' : train_f1_score,
                'Test f1' : test_f1_score
            }
        )

        # Calculate overfit/underfit variance gap   
        score_df['Gap'] = (score_df['Train f1'] - score_df['Test f1']).abs()

        # Filter out models with a variance gap greater than 5%
        safe_models_score = score_df[score_df['Gap'] <= 0.05]

        # If no model satisfies the gap condition , use all models.
        if safe_models_score.empty:
            report = score_df.sort_values( by="Test f1", ascending=False).reset_index(drop=True)
        else:
            report = safe_models_score.sort_values( by="Test f1", ascending=False).reset_index(drop=True)

        # Return report and trained models.
        return report, trained_models

    except Exception as e:
        # Raise a custom exception with detailed debugging information.
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try : 
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)


def tune_model(model, params, x_train, y_train):
    """
    Performs RandomizedSearchCV and returns the best estimator.
    """

    try:

        # Perform Randomized Search.
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=20,
            scoring="f1",
            cv=5,
            random_state=42,
            n_jobs=-1
        )

        # Log search configuration.
        mlflow.log_param("search_type", "RandomizedSearchCV")
        mlflow.log_param("n_iter", 20)
        mlflow.log_param("cv", 5)

        # Record tuning start time.
        start_time = time.time()

        # Train the model.
        random_search.fit(x_train, y_train)

        # Record tuning end time.
        end_time = time.time()

        # Log tuning time.
        mlflow.log_metric(
            "tuning_time_seconds",
            end_time - start_time
        )

        # Log best hyperparameters.
        mlflow.log_params(random_search.best_params_)

        # Log best cross-validation score.
        mlflow.log_metric(
            "best_cv_f1",
            random_search.best_score_
        )

        # Return the best trained model.
        return (
            random_search.best_estimator_,
            random_search.best_params_,
            random_search.best_score_
        )

    except Exception as e:
        raise CustomException(e, sys)