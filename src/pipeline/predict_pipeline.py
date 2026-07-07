import sys

import pandas as pd

from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        # Initialize the prediction pipeline.
        pass

    def predict(self, features):
        try:
            logging.info("Prediction pipeline started.")

            # Path of the saved model, preprocessor and label encoder.
            model_path = "artifacts/model.pkl"
            preprocessor_path = "artifacts/preprocessor.pkl"
            label_encoder_path = "artifacts/label_encoder.pkl"

            logging.info("Loading model, preprocessor and label encoder.")

            # Load saved objects.
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            label_encoder = load_object(label_encoder_path)

            # Apply preprocessing.
            data_scaled = preprocessor.transform(features)

            logging.info("Making prediction.")

            # Predict booking status , Convert prediction to integer.
            prediction = model.predict(data_scaled).astype(int)     

            # Convert numerical prediction back to original label.
            prediction = label_encoder.inverse_transform(prediction)

            logging.info(f"Prediction completed successfully. Predicted value: {prediction[0]}")

            return prediction[0]

        except Exception as e:
            logging.error("Error occurred during prediction.")
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        no_of_adults,
        no_of_children,
        no_of_weekend_nights,
        no_of_week_nights,
        type_of_meal_plan,
        required_car_parking_space,
        room_type_reserved,
        lead_time,
        arrival_month,
        market_segment_type,
        repeated_guest,
        no_of_previous_cancellations,
        avg_price_per_room,
        no_of_special_requests,
    ):
        # Store all user input values.
        self.no_of_adults = no_of_adults
        self.no_of_children = no_of_children
        self.no_of_weekend_nights = no_of_weekend_nights
        self.no_of_week_nights = no_of_week_nights
        self.type_of_meal_plan = type_of_meal_plan
        self.required_car_parking_space = required_car_parking_space
        self.room_type_reserved = room_type_reserved
        self.lead_time = lead_time
        self.arrival_month = arrival_month
        self.market_segment_type = market_segment_type
        self.repeated_guest = repeated_guest
        self.no_of_previous_cancellations = no_of_previous_cancellations
        self.avg_price_per_room = avg_price_per_room
        self.no_of_special_requests = no_of_special_requests

    def get_data_as_data_frame(self):
        try:
            logging.info("Creating DataFrame from user input.")

            # Store user inputs in dictionary format.
            custom_data_input_dict = {
                "no_of_adults": [self.no_of_adults],
                "no_of_children": [self.no_of_children],
                "no_of_weekend_nights": [self.no_of_weekend_nights],
                "no_of_week_nights": [self.no_of_week_nights],
                "type_of_meal_plan": [self.type_of_meal_plan],
                "required_car_parking_space": [self.required_car_parking_space],
                "room_type_reserved": [self.room_type_reserved],
                "lead_time": [self.lead_time],
                "arrival_month": [self.arrival_month],
                "market_segment_type": [self.market_segment_type],
                "repeated_guest": [self.repeated_guest],
                "no_of_previous_cancellations": [self.no_of_previous_cancellations],
                "avg_price_per_room": [self.avg_price_per_room],
                "no_of_special_requests": [self.no_of_special_requests],
            }

            # Convert dictionary into DataFrame.
            df = pd.DataFrame(custom_data_input_dict)

            logging.info("Input DataFrame created successfully.")

            return df

        except Exception as e:
            logging.error("Error occurred while creating DataFrame.")
            raise CustomException(e, sys)