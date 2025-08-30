#!/usr/bin/env python3
"""
Complete Fraud Detection Automation Script

This script provides end-to-end fraud detection automation:
1. Loads and preprocesses training data
2. Engineers features
3. Trains and saves XGBoost model
4. Loads new incoming transactions
5. Predicts fraud probability
6. Sends alerts (email/Slack) if fraud detected

Author: Fraud Detection System
Date: 2024
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
import logging
import argparse
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import components from previous modules
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

# Machine Learning imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FraudDetectionPipeline:
    """
    Complete fraud detection pipeline with automated alerts
    """
    
    def __init__(self, config_file='config.json'):
        """
        Initialize the fraud detection pipeline
        
        Args:
            config_file: Path to configuration file
        """
        self.config = self.load_config(config_file)
        self.model = None
        self.scaler = None
        self.feature_columns = None
        
        # Create necessary directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('logs', exist_ok=True)
        os.makedirs('data', exist_ok=True)
        
        logger.info("Fraud Detection Pipeline initialized")
    
    def load_config(self, config_file):
        """Load configuration from JSON file"""
        default_config = {
            "data": {
                "training_file": "creditcard.csv",
                "incoming_file": "new_transactions.csv",
                "test_size": 0.2,
                "random_state": 42
            },
            "model": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8
            },
            "alerts": {
                "fraud_threshold": 0.7,
                "email_enabled": False,
                "slack_enabled": False,
                "email_config": {
                    "smtp_server": "smtp.gmail.com",
                    "smtp_port": 587,
                    "sender_email": "",
                    "sender_password": "",
                    "recipient_email": "",
                    "cc_emails": []
                },
                "slack_config": {
                    "bot_token": "",
                    "channel_id": "#fraud-alerts",
                    "webhook_url": ""
                }
            },
            "logging": {
                "log_level": "INFO",
                "log_file": "fraud_detection.log"
            }
        }
        
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                # Merge with defaults
                for key in default_config:
                    if key not in config:
                        config[key] = default_config[key]
                return config
        except FileNotFoundError:
            logger.warning(f"Config file {config_file} not found. Using defaults.")
            return default_config
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON in {config_file}. Using defaults.")
            return default_config
    
    def load_and_preprocess_data(self, data_file=None):
        """
        Load and preprocess training data
        
        Args:
            data_file: Path to data file (optional)
        
        Returns:
            Preprocessed features and target
        """
        
        if data_file is None:
            data_file = self.config['data']['training_file']
        
        logger.info(f"Loading data from {data_file}")
        
        try:
            df = pd.read_csv(data_file)
            logger.info(f"Data loaded successfully: {df.shape}")
        except FileNotFoundError:
            logger.warning(f"Training file {data_file} not found. Creating sample data.")
            df = self.create_sample_data()
        
        # Basic EDA
        logger.info("Performing basic EDA...")
        fraud_count = df['Class'].sum()
        total_count = len(df)
        fraud_rate = fraud_count / total_count
        
        logger.info(f"Total transactions: {total_count}")
        logger.info(f"Fraud transactions: {fraud_count}")
        logger.info(f"Fraud rate: {fraud_rate:.4f}")
        
        # Check for missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            logger.warning(f"Missing values found: {missing_values}")
            df = df.dropna()
        
        return df
    
    def create_sample_data(self, n_samples=10000):
        """Create sample data for demonstration"""
        logger.info(f"Creating sample dataset with {n_samples} samples")
        
        np.random.seed(42)
        
        # Create sample data similar to creditcard.csv structure
        data = {}
        
        # Time feature
        data['Time'] = np.sort(np.random.randint(0, 172800, n_samples))
        
        # V1-V28 features (PCA components)
        for i in range(1, 29):
            data[f'V{i}'] = np.random.normal(0, 1, n_samples)
        
        # Amount feature
        data['Amount'] = np.random.lognormal(3, 1.5, n_samples)
        
        # Class (target) - highly imbalanced
        data['Class'] = np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
        
        # Make some transactions more fraudulent
        fraud_indices = np.where(data['Class'] == 1)[0]
        for idx in fraud_indices:
            data['Amount'][idx] = np.random.uniform(1000, 5000)
            data['V1'][idx] = np.random.normal(2, 1)
            data['V2'][idx] = np.random.normal(-2, 1)
        
        df = pd.DataFrame(data)
        logger.info("Sample data created successfully")
        return df
    
    def engineer_features(self, df):
        """
        Engineer additional features for fraud detection
        
        Args:
            df: Input dataframe
        
        Returns:
            DataFrame with engineered features
        """
        
        logger.info("Engineering features...")
        
        df_engineered = df.copy()
        
        # Create synthetic user IDs (in practice, you'd have real user IDs)
        np.random.seed(42)
        n_users = len(df) // 50
        df_engineered['User_ID'] = np.random.randint(1, n_users + 1, size=len(df))
        
        # Sort by User_ID and Time for proper calculations
        df_engineered = df_engineered.sort_values(['User_ID', 'Time'])
        
        # Feature 1: Transaction frequency per user
        user_freq = df_engineered.groupby('User_ID').size()
        df_engineered['Transaction_Frequency'] = df_engineered['User_ID'].map(user_freq)
        
        # Feature 2: Average transaction amount per user
        user_avg_amount = df_engineered.groupby('User_ID')['Amount'].mean()
        df_engineered['User_Avg_Amount'] = df_engineered['User_ID'].map(user_avg_amount)
        
        # Feature 3: Time since last transaction
        df_engineered['Time_Since_Last_Transaction'] = df_engineered.groupby('User_ID')['Time'].diff()
        median_time_diff = df_engineered['Time_Since_Last_Transaction'].median()
        df_engineered['Time_Since_Last_Transaction'].fillna(median_time_diff, inplace=True)
        
        # Feature 4: Amount deviation from user average
        df_engineered['Amount_Deviation_From_User_Avg'] = (
            df_engineered['Amount'] - df_engineered['User_Avg_Amount']
        ) / (df_engineered['User_Avg_Amount'] + 1e-8)
        
        # Feature 5: Transaction velocity
        user_time_range = df_engineered.groupby('User_ID')['Time'].apply(lambda x: x.max() - x.min() + 1)
        df_engineered['User_Time_Range'] = df_engineered['User_ID'].map(user_time_range)
        df_engineered['Transaction_Velocity'] = df_engineered['Transaction_Frequency'] / df_engineered['User_Time_Range']
        
        # Feature 6: Hour of day
        df_engineered['Hour_of_Day'] = (df_engineered['Time'] % 86400) // 3600
        
        # Feature 7: Day of week
        df_engineered['Day_of_Week'] = (df_engineered['Time'] // 86400) % 7
        
        # Feature 8: Amount percentile within user transactions
        df_engineered['Amount_Percentile_User'] = df_engineered.groupby('User_ID')['Amount'].rank(pct=True)
        
        # Replace infinite values
        df_engineered = df_engineered.replace([np.inf, -np.inf], [1e10, -1e10])
        
        logger.info(f"Feature engineering complete. New shape: {df_engineered.shape}")
        
        return df_engineered
    
    def train_model(self, df):
        """
        Train XGBoost model with feature engineering and class balancing
        
        Args:
            df: Training dataframe
        
        Returns:
            Trained model, scaler, and evaluation metrics
        """
        
        logger.info("Training fraud detection model...")
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['Class', 'User_ID']]
        X = df[feature_cols]
        y = df['Class']
        
        self.feature_columns = feature_cols
        logger.info(f"Training with {len(feature_cols)} features")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['data']['test_size'],
            random_state=self.config['data']['random_state'],
            stratify=y
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Calculate scale_pos_weight for XGBoost
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")
        
        # Train XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=self.config['model']['n_estimators'],
            max_depth=self.config['model']['max_depth'],
            learning_rate=self.config['model']['learning_rate'],
            subsample=self.config['model']['subsample'],
            colsample_bytree=self.config['model']['colsample_bytree'],
            scale_pos_weight=scale_pos_weight,
            random_state=self.config['data']['random_state'],
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        logger.info("Training XGBoost model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        logger.info("Model training completed!")
        logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
        logger.info("\nClassification Report:")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        metrics = {
            'roc_auc': roc_auc,
            'n_features': len(feature_cols),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        return metrics
    
    def save_model(self, model_name=None):
        """Save trained model and scaler"""
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"fraud_model_{timestamp}"
        
        model_path = f"models/{model_name}.joblib"
        scaler_path = f"models/{model_name}_scaler.joblib"
        
        # Save model and scaler
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'feature_columns': self.feature_columns,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'timestamp': datetime.now().isoformat(),
            'config': self.config
        }
        
        metadata_path = f"models/{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        logger.info(f"Model saved: {model_path}")
        logger.info(f"Scaler saved: {scaler_path}")
        logger.info(f"Metadata saved: {metadata_path}")
        
        return model_path, scaler_path, metadata_path
    
    def load_model(self, model_path, scaler_path, metadata_path=None):
        """Load trained model and scaler"""
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            if metadata_path and os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.feature_columns = metadata.get('feature_columns')
            
            logger.info(f"Model loaded from: {model_path}")
            logger.info(f"Scaler loaded from: {scaler_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def predict_transactions(self, transactions_df):
        """
        Predict fraud for new transactions
        
        Args:
            transactions_df: DataFrame with new transactions
        
        Returns:
            DataFrame with predictions
        """
        
        if self.model is None or self.scaler is None:
            logger.error("Model not loaded. Please load or train a model first.")
            return None
        
        logger.info(f"Predicting fraud for {len(transactions_df)} transactions")
        
        try:
            # Engineer features for new transactions
            df_engineered = self.engineer_features(transactions_df)
            
            # Prepare features (ensure same columns as training)
            if self.feature_columns:
                missing_cols = set(self.feature_columns) - set(df_engineered.columns)
                if missing_cols:
                    logger.warning(f"Missing columns: {missing_cols}")
                    # Add missing columns with default values
                    for col in missing_cols:
                        df_engineered[col] = 0
                
                # Select only the required features
                X = df_engineered[self.feature_columns]
            else:
                # Use all numeric columns except target
                feature_cols = [col for col in df_engineered.columns 
                              if col not in ['Class', 'User_ID'] and df_engineered[col].dtype in ['int64', 'float64']]
                X = df_engineered[feature_cols]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make predictions
            predictions = self.model.predict(X_scaled)
            prediction_probas = self.model.predict_proba(X_scaled)[:, 1]
            
            # Add predictions to results
            results_df = transactions_df.copy()
            results_df['Fraud_Prediction'] = predictions
            results_df['Fraud_Probability'] = prediction_probas
            results_df['Risk_Level'] = pd.cut(
                prediction_probas,
                bins=[0, 0.3, 0.7, 0.9, 1.0],
                labels=['LOW', 'MEDIUM', 'HIGH', 'CRITICAL']
            )
            
            logger.info(f"Predictions completed. Fraud detected: {predictions.sum()}")
            
            return results_df
            
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return None
    
    def send_email_alert(self, transaction_details):
        """Send email alert for fraud detection"""
        
        if not self.config['alerts']['email_enabled']:
            return False
        
        email_config = self.config['alerts']['email_config']
        
        try:
            # Create email message
            message = MIMEMultipart("alternative")
            message["Subject"] = f"🚨 FRAUD ALERT - Transaction {transaction_details.get('transaction_id', 'Unknown')}"
            message["From"] = email_config['sender_email']
            message["To"] = email_config['recipient_email']
            
            # Email content
            txn_id = transaction_details.get('transaction_id', 'Unknown')
            amount = transaction_details.get('amount', 0)
            probability = transaction_details.get('fraud_probability', 0)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            text_content = f"""
            🚨 FRAUD DETECTION ALERT 🚨
            
            Transaction Details:
            - Transaction ID: {txn_id}
            - Amount: ${amount:.2f}
            - Fraud Probability: {probability:.2%}
            - Detection Time: {timestamp}
            
            Please investigate immediately.
            """
            
            text_part = MIMEText(text_content, "plain")
            message.attach(text_part)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(email_config['smtp_server'], email_config['smtp_port']) as server:
                server.starttls(context=context)
                server.login(email_config['sender_email'], email_config['sender_password'])
                server.sendmail(
                    email_config['sender_email'], 
                    email_config['recipient_email'], 
                    message.as_string()
                )
            
            logger.info(f"Email alert sent for transaction {txn_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {str(e)}")
            return False
    
    def send_slack_alert(self, transaction_details):
        """Send Slack alert for fraud detection"""
        
        if not self.config['alerts']['slack_enabled']:
            return False
        
        slack_config = self.config['alerts']['slack_config']
        webhook_url = slack_config.get('webhook_url')
        
        if not webhook_url:
            logger.error("Slack webhook URL not configured")
            return False
        
        try:
            txn_id = transaction_details.get('transaction_id', 'Unknown')
            amount = transaction_details.get('amount', 0)
            probability = transaction_details.get('fraud_probability', 0)
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Create Slack payload
            payload = {
                "text": f"🚨 FRAUD ALERT: Transaction {txn_id}",
                "attachments": [
                    {
                        "color": "#FF0000" if probability >= 0.9 else "#FF6600",
                        "title": "🚨 Fraud Detection Alert",
                        "fields": [
                            {"title": "Transaction ID", "value": txn_id, "short": True},
                            {"title": "Amount", "value": f"${amount:,.2f}", "short": True},
                            {"title": "Fraud Probability", "value": f"{probability:.1%}", "short": True},
                            {"title": "Detection Time", "value": timestamp, "short": True}
                        ],
                        "footer": "Fraud Detection System",
                        "ts": int(datetime.now().timestamp())
                    }
                ]
            }
            
            # Send webhook
            response = requests.post(webhook_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                logger.info(f"Slack alert sent for transaction {txn_id}")
                return True
            else:
                logger.error(f"Slack webhook failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {str(e)}")
            return False
    
    def process_alerts(self, fraud_transactions):
        """Process alerts for fraud transactions"""
        
        fraud_threshold = self.config['alerts']['fraud_threshold']
        alert_count = 0
        
        for _, row in fraud_transactions.iterrows():
            if row['Fraud_Probability'] > fraud_threshold:
                transaction_details = {
                    'transaction_id': row.get('Transaction_ID', f"TXN_{len(fraud_transactions)}"),
                    'amount': row.get('Amount', 0),
                    'fraud_probability': row['Fraud_Probability']
                }
                
                # Send email alert
                if self.config['alerts']['email_enabled']:
                    self.send_email_alert(transaction_details)
                
                # Send Slack alert
                if self.config['alerts']['slack_enabled']:
                    self.send_slack_alert(transaction_details)
                
                alert_count += 1
        
        logger.info(f"Sent {alert_count} fraud alerts")
        return alert_count
    
    def run_full_pipeline(self, training_file=None, incoming_file=None, retrain=True):
        """
        Run the complete fraud detection pipeline
        
        Args:
            training_file: Path to training data file
            incoming_file: Path to incoming transactions file
            retrain: Whether to retrain the model
        
        Returns:
            Results dictionary
        """
        
        logger.info("=" * 60)
        logger.info("STARTING FRAUD DETECTION AUTOMATION PIPELINE")
        logger.info("=" * 60)
        
        results = {
            'training_completed': False,
            'model_saved': False,
            'predictions_made': False,
            'alerts_sent': 0,
            'fraud_detected': 0
        }
        
        try:
            # Step 1: Load and preprocess training data
            if retrain:
                logger.info("Step 1: Loading and preprocessing training data...")
                df = self.load_and_preprocess_data(training_file)
                
                # Step 2: Engineer features
                logger.info("Step 2: Engineering features...")
                df_engineered = self.engineer_features(df)
                
                # Step 3: Train model
                logger.info("Step 3: Training model...")
                metrics = self.train_model(df_engineered)
                results['training_completed'] = True
                results['model_metrics'] = metrics
                
                # Step 4: Save model
                logger.info("Step 4: Saving model...")
                model_paths = self.save_model()
                results['model_saved'] = True
                results['model_paths'] = model_paths
            
            else:
                # Load existing model
                logger.info("Loading existing model...")
                # This would load the most recent model
                # Implementation depends on your model versioning strategy
                pass
            
            # Step 5: Load incoming transactions
            if incoming_file is None:
                incoming_file = self.config['data']['incoming_file']
            
            logger.info(f"Step 5: Loading incoming transactions from {incoming_file}...")
            
            try:
                new_transactions = pd.read_csv(incoming_file)
                logger.info(f"Loaded {len(new_transactions)} new transactions")
            except FileNotFoundError:
                logger.warning(f"Incoming file {incoming_file} not found. Creating sample transactions.")
                new_transactions = self.create_sample_incoming_transactions()
            
            # Step 6: Make predictions
            logger.info("Step 6: Making fraud predictions...")
            predictions_df = self.predict_transactions(new_transactions)
            
            if predictions_df is not None:
                results['predictions_made'] = True
                
                # Step 7: Process fraud alerts
                logger.info("Step 7: Processing fraud alerts...")
                fraud_transactions = predictions_df[
                    predictions_df['Fraud_Probability'] > self.config['alerts']['fraud_threshold']
                ]
                
                results['fraud_detected'] = len(fraud_transactions)
                
                if len(fraud_transactions) > 0:
                    logger.info(f"Found {len(fraud_transactions)} suspicious transactions")
                    
                    # Send alerts
                    alert_count = self.process_alerts(fraud_transactions)
                    results['alerts_sent'] = alert_count
                    
                    # Save fraud report
                    self.save_fraud_report(fraud_transactions, predictions_df)
                
                else:
                    logger.info("No fraudulent transactions detected")
            
            # Step 8: Generate summary report
            logger.info("Step 8: Generating summary report...")
            self.generate_summary_report(results, predictions_df)
            
            logger.info("=" * 60)
            logger.info("FRAUD DETECTION PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def create_sample_incoming_transactions(self, n_transactions=50):
        """Create sample incoming transactions for testing"""
        
        logger.info(f"Creating {n_transactions} sample incoming transactions")
        
        np.random.seed(int(datetime.now().timestamp()) % 2**32)
        
        transactions = []
        for i in range(n_transactions):
            txn = {
                'Transaction_ID': f'TXN_{datetime.now().strftime("%Y%m%d")}_{i+1:04d}',
                'Time': np.random.randint(0, 86400),  # Random time in a day
                'Amount': np.random.lognormal(3, 1.5)
            }
            
            # Add V features
            for j in range(1, 29):
                txn[f'V{j}'] = np.random.normal(0, 1)
            
            # Make some transactions suspicious (10% chance)
            if np.random.random() < 0.1:
                txn['Amount'] = np.random.uniform(2000, 10000)
                txn['V1'] = np.random.normal(3, 1)
                txn['V2'] = np.random.normal(-3, 1)
            
            transactions.append(txn)
        
        df = pd.DataFrame(transactions)
        
        # Save to file for future use
        incoming_file = self.config['data']['incoming_file']
        df.to_csv(incoming_file, index=False)
        logger.info(f"Sample transactions saved to {incoming_file}")
        
        return df
    
    def save_fraud_report(self, fraud_transactions, all_predictions):
        """Save detailed fraud report"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"logs/fraud_report_{timestamp}.csv"
        
        # Save fraud transactions
        fraud_transactions.to_csv(report_file, index=False)
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_transactions': len(all_predictions),
            'fraud_detected': len(fraud_transactions),
            'fraud_rate': len(fraud_transactions) / len(all_predictions),
            'avg_fraud_probability': fraud_transactions['Fraud_Probability'].mean() if len(fraud_transactions) > 0 else 0,
            'total_amount_at_risk': fraud_transactions['Amount'].sum() if len(fraud_transactions) > 0 else 0
        }
        
        summary_file = f"logs/fraud_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=4)
        
        logger.info(f"Fraud report saved: {report_file}")
        logger.info(f"Summary saved: {summary_file}")
    
    def generate_summary_report(self, results, predictions_df):
        """Generate and log summary report"""
        
        logger.info("\n" + "=" * 50)
        logger.info("PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 50)
        
        if predictions_df is not None:
            logger.info(f"Total transactions processed: {len(predictions_df)}")
            logger.info(f"Fraud transactions detected: {results['fraud_detected']}")
            logger.info(f"Fraud rate: {results['fraud_detected']/len(predictions_df):.4f}")
            logger.info(f"Alerts sent: {results['alerts_sent']}")
            
            if results['fraud_detected'] > 0:
                avg_fraud_prob = predictions_df[predictions_df['Fraud_Probability'] > self.config['alerts']['fraud_threshold']]['Fraud_Probability'].mean()
                total_risk_amount = predictions_df[predictions_df['Fraud_Probability'] > self.config['alerts']['fraud_threshold']]['Amount'].sum()
                logger.info(f"Average fraud probability: {avg_fraud_prob:.4f}")
                logger.info(f"Total amount at risk: ${total_risk_amount:,.2f}")
        
        logger.info(f"Training completed: {results['training_completed']}")
        logger.info(f"Model saved: {results['model_saved']}")
        logger.info(f"Predictions made: {results['predictions_made']}")
        
        if 'model_metrics' in results:
            logger.info(f"Model ROC-AUC: {results['model_metrics']['roc_auc']:.4f}")
        
        logger.info("=" * 50)

def create_sample_config():
    """Create a sample configuration file"""
    
    config = {
        "data": {
            "training_file": "creditcard.csv",
            "incoming_file": "new_transactions.csv",
            "test_size": 0.2,
            "random_state": 42
        },
        "model": {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8
        },
        "alerts": {
            "fraud_threshold": 0.7,
            "email_enabled": True,
            "slack_enabled": True,
            "email_config": {
                "smtp_server": "smtp.gmail.com",
                "smtp_port": 587,
                "sender_email": "your_email@gmail.com",
                "sender_password": "your_app_password",
                "recipient_email": "fraud_team@company.com",
                "cc_emails": ["security@company.com", "manager@company.com"]
            },
            "slack_config": {
                "bot_token": "xoxb-your-bot-token",
                "channel_id": "#fraud-alerts",
                "webhook_url": "https://hooks.slack.com/services/your/webhook/url"
            }
        },
        "logging": {
            "log_level": "INFO",
            "log_file": "fraud_detection.log"
        }
    }
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    print("Sample configuration file 'config.json' created.")
    print("Please update with your actual credentials and settings.")

def main():
    """Main function for command line interface"""
    
    parser = argparse.ArgumentParser(description='Fraud Detection Automation Pipeline')
    parser.add_argument('--config', default='config.json', help='Configuration file path')
    parser.add_argument('--training-file', help='Training data file path')
    parser.add_argument('--incoming-file', help='Incoming transactions file path')
    parser.add_argument('--no-retrain', action='store_true', help='Skip model retraining')
    parser.add_argument('--create-config', action='store_true', help='Create sample configuration file')
    parser.add_argument('--test-mode', action='store_true', help='Run in test mode with sample data')
    
    args = parser.parse_args()
    
    if args.create_config:
        create_sample_config()
        return
    
    # Initialize pipeline
    pipeline = FraudDetectionPipeline(args.config)
    
    if args.test_mode:
        logger.info("Running in test mode with sample data")
        # Disable actual alerts in test mode
        pipeline.config['alerts']['email_enabled'] = False
        pipeline.config['alerts']['slack_enabled'] = False
    
    # Run the pipeline
    try:
        results = pipeline.run_full_pipeline(
            training_file=args.training_file,
            incoming_file=args.incoming_file,
            retrain=not args.no_retrain
        )
        
        if 'error' in results:
            logger.error(f"Pipeline execution failed: {results['error']}")
            exit(1)
        else:
            logger.info("Pipeline execution completed successfully!")
            
            # Print key results
            print("\n" + "="*60)
            print("EXECUTION RESULTS")
            print("="*60)
            print(f"Fraud transactions detected: {results.get('fraud_detected', 0)}")
            print(f"Alerts sent: {results.get('alerts_sent', 0)}")
            print(f"Training completed: {results.get('training_completed', False)}")
            print(f"Model saved: {results.get('model_saved', False)}")
            print("="*60)
    
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        exit(1)

def demo_pipeline():
    """Demonstration of the complete pipeline"""
    
    print("FRAUD DETECTION AUTOMATION PIPELINE DEMO")
    print("=" * 60)
    
    # Create demo pipeline
    pipeline = FraudDetectionPipeline()
    
    # Disable alerts for demo
    pipeline.config['alerts']['email_enabled'] = False
    pipeline.config['alerts']['slack_enabled'] = False
    pipeline.config['alerts']['fraud_threshold'] = 0.5  # Lower threshold for demo
    
    print("\n1. Pipeline Configuration:")
    print(f"   - Fraud threshold: {pipeline.config['alerts']['fraud_threshold']}")
    print(f"   - Email alerts: {pipeline.config['alerts']['email_enabled']}")
    print(f"   - Slack alerts: {pipeline.config['alerts']['slack_enabled']}")
    
    print("\n2. Running complete pipeline...")
    
    try:
        # Run pipeline
        results = pipeline.run_full_pipeline(retrain=True)
        
        print("\n3. Demo Results:")
        print(f"   - Training completed: ✓" if results['training_completed'] else "   - Training completed: ✗")
        print(f"   - Model saved: ✓" if results['model_saved'] else "   - Model saved: ✗")
        print(f"   - Predictions made: ✓" if results['predictions_made'] else "   - Predictions made: ✗")
        print(f"   - Fraud detected: {results['fraud_detected']}")
        print(f"   - Alerts sent: {results['alerts_sent']}")
        
        if 'model_metrics' in results:
            print(f"   - Model ROC-AUC: {results['model_metrics']['roc_auc']:.4f}")
        
        print("\n4. Files Created:")
        print("   - Model files in 'models/' directory")
        print("   - Log files in 'logs/' directory")
        print("   - Sample data in 'data/' directory")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")
        
    return pipeline

if __name__ == "__main__":
    import sys
    
    # Check if running as demo
    if len(sys.argv) == 1 or (len(sys.argv) == 2 and sys.argv[1] == 'demo'):
        print("Running in demo mode...")
        pipeline = demo_pipeline()
    else:
        main()

# Additional utility functions for advanced usage

class FraudMonitor:
    """Real-time fraud monitoring class"""
    
    def __init__(self, pipeline, monitoring_interval=60):
        """
        Initialize fraud monitor
        
        Args:
            pipeline: FraudDetectionPipeline instance
            monitoring_interval: Monitoring interval in seconds
        """
        self.pipeline = pipeline
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
    
    def start_monitoring(self, incoming_file):
        """Start continuous monitoring of incoming transactions"""
        
        import time
        
        self.is_monitoring = True
        logger.info(f"Starting fraud monitoring with {self.monitoring_interval}s interval")
        
        try:
            while self.is_monitoring:
                # Check for new transactions
                try:
                    new_transactions = pd.read_csv(incoming_file)
                    
                    if len(new_transactions) > 0:
                        logger.info(f"Processing {len(new_transactions)} new transactions")
                        
                        # Make predictions
                        predictions_df = self.pipeline.predict_transactions(new_transactions)
                        
                        if predictions_df is not None:
                            # Check for fraud
                            fraud_transactions = predictions_df[
                                predictions_df['Fraud_Probability'] > self.pipeline.config['alerts']['fraud_threshold']
                            ]
                            
                            if len(fraud_transactions) > 0:
                                logger.warning(f"FRAUD DETECTED: {len(fraud_transactions)} suspicious transactions")
                                
                                # Send alerts
                                self.pipeline.process_alerts(fraud_transactions)
                            
                            # Clear processed transactions (in practice, you'd move them to processed folder)
                            # os.remove(incoming_file)
                    
                except FileNotFoundError:
                    # No new transactions
                    pass
                except Exception as e:
                    logger.error(f"Monitoring error: {str(e)}")
                
                # Wait for next interval
                time.sleep(self.monitoring_interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            self.is_monitoring = False

def batch_process_files(pipeline, input_directory, output_directory):
    """Batch process multiple transaction files"""
    
    import glob
    
    os.makedirs(output_directory, exist_ok=True)
    
    # Find all CSV files in input directory
    csv_files = glob.glob(os.path.join(input_directory, "*.csv"))
    
    logger.info(f"Found {len(csv_files)} files to process")
    
    results = []
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        logger.info(f"Processing {filename}...")
        
        try:
            # Load transactions
            transactions = pd.read_csv(file_path)
            
            # Make predictions
            predictions_df = pipeline.predict_transactions(transactions)
            
            if predictions_df is not None:
                # Save results
                output_path = os.path.join(output_directory, f"predictions_{filename}")
                predictions_df.to_csv(output_path, index=False)
                
                # Count fraud
                fraud_count = (predictions_df['Fraud_Probability'] > pipeline.config['alerts']['fraud_threshold']).sum()
                
                results.append({
                    'file': filename,
                    'total_transactions': len(predictions_df),
                    'fraud_detected': fraud_count,
                    'output_file': output_path
                })
                
                logger.info(f"✓ {filename}: {fraud_count} fraud detected out of {len(predictions_df)}")
            
        except Exception as e:
            logger.error(f"✗ Failed to process {filename}: {str(e)}")
            results.append({
                'file': filename,
                'error': str(e)
            })
    
    # Save batch results summary
    summary_path = os.path.join(output_directory, "batch_summary.json")
    with open(summary_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'files_processed': len(results),
            'results': results
        }, f, indent=4)
    
    logger.info(f"Batch processing complete. Summary saved to {summary_path}")
    return results

# Example usage and documentation
USAGE_EXAMPLES = """
FRAUD DETECTION AUTOMATION SCRIPT USAGE EXAMPLES:

1. Basic Usage (Demo Mode):
   python automation_script.py demo

2. Create Configuration File:
   python automation_script.py --create-config

3. Run with Custom Config:
   python automation_script.py --config my_config.json

4. Run with Custom Data Files:
   python automation_script.py --training-file data/training.csv --incoming-file data/new_txns.csv

5. Run Without Retraining (Use Existing Model):
   python automation_script.py --no-retrain

6. Test Mode (No Real Alerts):
   python automation_script.py --test-mode

CONFIGURATION:
- Edit config.json to set up email/Slack credentials
- Adjust fraud threshold and model parameters
- Configure data file paths

DIRECTORY STRUCTURE CREATED:
- models/     : Saved models and scalers
- logs/       : Execution logs and fraud reports  
- data/       : Sample data files

REQUIREMENTS:
- pandas, numpy, scikit-learn, xgboost
- For email: smtplib (built-in)
- For Slack: requests (for webhooks) or slack_sdk

SECURITY NOTES:
- Store credentials securely (environment variables recommended)
- Use app passwords for Gmail, not regular passwords
- Regularly update and retrain models
- Monitor alert logs for compliance

For more information, see the documentation in each function.
"""

print(__doc__ if __name__ == "__main__" else "", end="")