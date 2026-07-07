---
title: Hotel Reservation Predictor
emoji: 🏨
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
---

# 🏨 Hotel Reservation Prediction Pipeline

An End-to-End Machine Learning project that predicts whether a hotel reservation will be canceled or honored based on customer data. This project features a full MLOps pipeline including data ingestion, transformation, model training, MLflow experiment tracking via DagsHub, and a Dockerized Flask web application deployed on Hugging Face Spaces.

## 🚀 Live Demo
**[Insert your Hugging Face Space URL here]**

## 🏗️ Project Architecture & MLOps

* **Frontend:** HTML/CSS form for user input.
* **Backend:** Flask web server (`app.py`) routing inputs through the prediction pipeline.
* **Machine Learning Pipeline:** Modularized Python scripts for Data Ingestion, Data Transformation, and Model Training using Scikit-Learn.
* **Experiment Tracking:** MLflow integrated with **DagsHub** to track model metrics (F1 Score, RMSE), parameters, and artifacts in the cloud.
* **Large File Storage:** Git LFS is utilized to manage the heavy `model.pkl` artifacts.
* **Deployment:** Containerized using **Docker** and deployed on **Hugging Face Spaces**.

## 💻 Tech Stack

* **Language:** Python 3.9
* **Libraries:** Pandas, Numpy, Scikit-Learn, Flask, MLflow, DagsHub
* **Server:** Gunicorn
* **Containerization:** Docker
* **Version Control:** Git & GitHub

## 📂 Project Structure

```text
├── artifacts/                # Saved models, preprocessor, and label encoder (.pkl files managed by Git LFS)
├── src/                      # Source code for the ML pipeline
│   ├── components/           # Data Ingestion, Transformation, and Model Trainer modules
│   ├── pipeline/             # Training and Prediction pipeline scripts
│   ├── exception.py          # Custom exception handling
│   ├── logger.py             # Custom logging functionality
│   └── utils.py              # Utility functions (e.g., model saving/loading)
├── templates/                # HTML files for the Flask frontend
├── static/                   # CSS/JS files
├── app.py                    # Flask application entry point
├── Dockerfile                # Docker configuration for Hugging Face deployment
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation