import os
import sys
import mlflow
from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier , AdaBoostClassifier , GradientBoostingClassifier)
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models

import warnings
warnings.filterwarnings("ignore")


# Stores the path where the trained model will be saved.
@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        # Create an object containing model configuration.
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            # Separate features and target variables.
            logging.info('Splitting Training and Test input data')

            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Convert target values to integer.
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

            # Dictionary containing all candidate regression models.
            models = {
                'Logistic Regression' : LogisticRegression() ,
                'Linear Support Vector Classifier' : LinearSVC() ,
                'Gaussian NB' : GaussianNB() , 
                'KNN Classifier' : KNeighborsClassifier() ,
                'Decision Tree Classifier' : DecisionTreeClassifier() , 
                'Random Forest Classifier' : RandomForestClassifier() ,
                'AdaBoost Classifier' : AdaBoostClassifier() ,
                'Gradient Boosting classifier' : GradientBoostingClassifier() , 
                'XGBoost Classifier' : XGBClassifier(),
                'CatBoost Classifier' : CatBoostClassifier(verbose=False) 
            }

            # Train and evaluate all models.
            model_report, trained_models = evaluate_models(
                x_train,
                y_train,
                x_test,
                y_test,
                models
            )

            # Select the model with the highest F1 score.
            best_model_name = model_report.iloc[0, 0]
            best_model_score = model_report.iloc[0, 2]

            best_model = trained_models[best_model_name]

            # Ensure the selected model meets the minimum performance threshold.
            if best_model_score < 0.60:
                raise CustomException("No suitable model found.", sys)

            logging.info(f'Best Model Found: {best_model_name}')

            # Save the best-performing model.
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # MLflow Integration Start: Log the Champion Model to the Parent Run
            if mlflow.active_run():
                logging.info("Logging best model and metrics to MLflow Parent Run")
                
                # Log the winning model's name and score so it's visible on the main dashboard
                mlflow.log_param("champion_model_name", best_model_name)
                mlflow.log_metric("champion_test_f1", best_model_score)
                
                # Log the actual final model artifact to the parent run for easy deployment
                mlflow.sklearn.log_model(
                    sk_model=best_model, 
                    name=best_model_name.replace(" ", "_"),
                    serialization_format="cloudpickle"
                )


            logging.info('Best model saved successfully.')

            # Evaluate the saved model on testing data.
            predicted = best_model.predict(x_test)

            f1 = f1_score(y_test, predicted)

            return best_model_name, f1

        except Exception as e:
            raise CustomException(e, sys)
      