import sys
from flask import Flask, render_template, request

from src.exception import CustomException
from src.logger import logging
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


# Create the Flask application.
app = Flask(__name__)


# Home page route.
@app.route('/')
def index():
    try:
        logging.info("Home page requested.")
        return render_template('index.html')

    except Exception as e:
        logging.error("Error while loading home page.")
        raise CustomException(e, sys)


# Route for prediction.
@app.route('/predictData', methods=['GET', 'POST'])
def predict_datapoint():

    try:
        # Display the prediction form.
        if request.method == 'GET':
            logging.info("Prediction page opened.")
            return render_template('home.html')

        logging.info("Received prediction request.")

        # Collect user input from the HTML form.
        data = CustomData(
            no_of_adults=int(request.form.get('no_of_adults')),
            no_of_children=int(request.form.get('no_of_children')),
            no_of_weekend_nights=int(request.form.get('no_of_weekend_nights')),
            no_of_week_nights=int(request.form.get('no_of_week_nights')),
            type_of_meal_plan=request.form.get('type_of_meal_plan'),
            required_car_parking_space=int(request.form.get('required_car_parking_space')),
            room_type_reserved=request.form.get('room_type_reserved'),
            lead_time=int(request.form.get('lead_time')),
            arrival_month=int(request.form.get('arrival_month')),
            market_segment_type=request.form.get('market_segment_type'),
            repeated_guest=int(request.form.get('repeated_guest')),
            no_of_previous_cancellations=int(request.form.get('no_of_previous_cancellations')),
            avg_price_per_room=float(request.form.get('avg_price_per_room')),
            no_of_special_requests=int(request.form.get('no_of_special_requests')),
        )

        logging.info("User input collected successfully.")

        # Convert user input into a DataFrame.
        pred_df = data.get_data_as_data_frame()

        logging.info("Input converted into DataFrame.")

        # Create the prediction pipeline.
        predict_pipeline = PredictPipeline()

        # Generate prediction.
        results = predict_pipeline.predict(pred_df)

        logging.info(f"Prediction generated successfully: {results}")

        # Display prediction on the webpage.
        return render_template(
            'home.html',
            results=results
        )

    except Exception as e:
        logging.error("Error occurred during prediction.")
        raise CustomException(e, sys)


# Start the Flask development server.
if __name__ == '__main__':
    logging.info("Starting Flask application.")

    app.run(host='0.0.0.0', port=7860) # Remove debug=True during deployment.