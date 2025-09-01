## Fraud Detection System
This project is a complete fraud detection pipeline that uses machine learning to identify fraudulent transactions.
It covers everything from exploratory data analysis (EDA) to model training, prediction, and automated alerts via email and Slack.

# Features
- Data Exploration & Visualization → Class balance, transaction distributions
- Feature Engineering → Time-based and behavioral features to improve model accuracy
- Model Training → Random Forest and XGBoost classifiers
- Model Saving & Prediction → Save trained models, scalers, and metadata for reuse
- Email Alert System → Get notified when fraud is detected
- Slack Alert System → Team notifications through Slack bots or webhooks
- End-to-End Pipeline → Automates the full fraud detection workflow

# Project Structure
FRAUD/
- │
- ├── data/                          # Dataset storage
- ├── logs/                          # Log files
- ├── models/                        # Trained models and scalers
- │
- ├── 1_data_analysis.py
- ├── 2_feature_engineering.py
- ├── 3_model_training.py
- ├── 4_model_saving_prediction.py
- ├── 5_alert_system.py
- ├── 6_slack_alert_system.py
- ├── 7_fraud_detection_pipeline.py
- │
- ├── creditcard.csv                 # Main dataset (download from Kaggle)
- ├── new_transactions.csv           # New unseen transactions for testing
- ├── Distributiongraphs.png         # Dataset visualizations
- ├── Graphs.png                     # Model comparison and performance metrics
- ├── fraud_detection.log            # Sample log file
- ├── requirements.txt               # Dependencies
- └── README.md                      # Documentation

# Dataset
We use the Credit Card Fraud Detection Dataset from Kaggle:
👉 https://www.kaggle.com/mlg-ulb/creditcardfraud

- Total transactions: 284,807
- Fraudulent transactions: 492 (~0.17%)
- Highly imbalanced dataset

You must download the dataset yourself and place creditcard.csv into the FRAUD/ folder.

# Exploratory Data Analysis
# Class Distribution
- Fraud cases are extremely rare compared to normal ones

# Model Performance
- Both Random Forest and XGBoost achieve high ROC-AUC (~0.98)
- Random Forest performs slightly better overall
- Confusion matrices show very few false negatives, which is critical in fraud detection

# Installation
Clone this repository:
- git clone https://github.com/sdlk4/Fraud-Detection-System.git
- cd FRAUD
  
Install dependencies:
- pip install -r requirements.txt

# Usage
Run the full pipeline
- python 7_fraud_detection_pipeline.py
  
This will:
- Load and preprocess dataset
- Engineer new features
- Train the fraud detection model
- Save model + scaler + metadata
- Predict on new transactions (new_transactions.csv)
- Trigger email/Slack alerts if fraud is found
- Generate visualizations and logs

# Run scripts individually
- Data analysis → python 1_data_analysis.py
- Feature engineering → python 2_feature_engineering.py
- Model training → python 3_model_training.py
- Model saving & prediction → python 4_model_saving_prediction.py
- Email alerts → python 5_alert_system.py
- Slack alerts → python 6_slack_alert_system.py

# Email Alerts
- Requires Gmail App Password (not your real password)
- Update credentials inside 5_alert_system.py
- Fraud alerts will be sent automatically when suspicious activity is detected

# Slack Alerts
- Configure with either a Bot Token or Webhook URL
- Update details inside 6_slack_alert_system.py

# Results
- ROC-AUC: ~0.978 (Random Forest), ~0.976 (XGBoost)
- Precision-Recall curves show good fraud detection performance despite imbalance
- Confusion Matrices confirm very few missed fraud cases

# Example prediction output:
- Transaction ID: 105
- Fraud Probability: 0.87 (87%)
- Prediction: FRAUD
- Email Alert: Sent
- Slack Alert: Sent

# Contributing
Found a bug or want to improve the system?
- Fork the repository
- Create a feature branch
- Make your changes
- Submit a pull request

# Technical Notes
- Uses SMOTE for handling imbalanced datasets
- Feature scaling applied automatically
- Model persistence with joblib for production deployment
- Comprehensive logging for monitoring and debugging
- Configurable alert thresholds based on business needs
