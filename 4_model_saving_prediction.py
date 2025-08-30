import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def save_model_and_scaler(model, scaler, model_name="xgboost_fraud_model"):
    """
    Save the trained XGBoost model and scaler using joblib
    """
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Generate timestamp for versioning
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define file paths
    model_path = f'models/{model_name}_{timestamp}.joblib'
    scaler_path = f'models/scaler_{model_name}_{timestamp}.joblib'
    
    try:
        # Save model
        joblib.dump(model, model_path)
        print(f" Model saved successfully: {model_path}")
        
        # Save scaler
        joblib.dump(scaler, scaler_path)
        print(f" Scaler saved successfully: {scaler_path}")
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'timestamp': timestamp,
            'model_path': model_path,
            'scaler_path': scaler_path,
            'model_type': str(type(model).__name__)
        }
        
        metadata_path = f'models/metadata_{model_name}_{timestamp}.json'
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        print(f" Metadata saved: {metadata_path}")
        
        return {
            'model_path': model_path,
            'scaler_path': scaler_path,
            'metadata_path': metadata_path
        }
        
    except Exception as e:
        print(f"❌ Error saving model: {str(e)}")
        return None

def load_model_and_scaler(model_path, scaler_path):
    """
    Load the trained model and scaler from saved files
    """
    
    try:
        # Load model
        model = joblib.load(model_path)
        print(f" Model loaded successfully from: {model_path}")
        
        # Load scaler
        scaler = joblib.load(scaler_path)
        print(f" Scaler loaded successfully from: {scaler_path}")
        
        print(f"Model type: {type(model).__name__}")
        
        return model, scaler
        
    except FileNotFoundError as e:
        print(f"❌ File not found: {str(e)}")
        return None, None
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return None, None

def predict_single_transaction(model, scaler, transaction_data, fraud_threshold=0.7):
    """
    Predict fraud for a single transaction
    
    Args:
        model: Trained fraud detection model
        scaler: Fitted StandardScaler
        transaction_data: Dictionary or pandas Series with transaction features
        fraud_threshold: Probability threshold for fraud detection (default 0.7)
    
    Returns:
        Dictionary with prediction results
    """
    
    try:
        # Convert to DataFrame if it's a dictionary
        if isinstance(transaction_data, dict):
            df = pd.DataFrame([transaction_data])
        elif isinstance(transaction_data, pd.Series):
            df = pd.DataFrame([transaction_data])
        else:
            df = transaction_data.copy()
        
        print("=== TRANSACTION ANALYSIS ===")
        print(f"Transaction ID: {df.get('Transaction_ID', ['Unknown']).iloc[0]}")
        print(f"Amount: ${df.get('Amount', ['Unknown']).iloc[0]}")
        print(f"Time: {df.get('Time', ['Unknown']).iloc[0]}")
        
        # Remove non-feature columns if present
        feature_columns = [col for col in df.columns if col not in ['Transaction_ID', 'Class']]
        X = df[feature_columns]
        
        print(f"Features used for prediction: {len(feature_columns)}")
        
        # Scale the features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        prediction_proba = model.predict_proba(X_scaled)[0]
        
        fraud_probability = prediction_proba[1]  # Probability of fraud (class 1)
        
        # Determine if fraud based on threshold
        is_fraud = fraud_probability > fraud_threshold
        
        print(f"\n=== PREDICTION RESULTS ===")
        print(f"Fraud Probability: {fraud_probability:.4f} ({fraud_probability*100:.2f}%)")
        print(f"Prediction: {'FRAUD' if prediction == 1 else 'NORMAL'}")
        print(f"Threshold-based Detection: {'FRAUD DETECTED!' if is_fraud else 'Normal Transaction'}")
        
        # Alert if fraud detected
        if is_fraud:
            print("\n" + "="*50)
            print("🚨 FRAUD DETECTED! 🚨")
            print("="*50)
            print(f"Transaction flagged as fraudulent!")
            print(f"Confidence: {fraud_probability*100:.2f}%")
            print(f"Threshold: {fraud_threshold*100:.0f}%")
        
        result = {
            'transaction_id': df.get('Transaction_ID', ['Unknown']).iloc[0],
            'amount': df.get('Amount', ['Unknown']).iloc[0],
            'prediction': int(prediction),
            'fraud_probability': float(fraud_probability),
            'is_fraud_detected': is_fraud,
            'threshold_used': fraud_threshold,
            'prediction_label': 'FRAUD' if prediction == 1 else 'NORMAL',
            'confidence_level': 'HIGH' if fraud_probability > 0.8 else 'MEDIUM' if fraud_probability > 0.5 else 'LOW'
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Error making prediction: {str(e)}")
        return None

def predict_batch_transactions(model, scaler, transactions_df, fraud_threshold=0.7):
    """
    Predict fraud for multiple transactions
    """
    
    try:
        print(f"=== BATCH PREDICTION FOR {len(transactions_df)} TRANSACTIONS ===")
        
        # Prepare features
        feature_columns = [col for col in transactions_df.columns if col not in ['Transaction_ID', 'Class']]
        X = transactions_df[feature_columns]
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make predictions
        predictions = model.predict(X_scaled)
        prediction_probas = model.predict_proba(X_scaled)[:, 1]
        
        # Create results dataframe
        results_df = transactions_df.copy()
        results_df['Fraud_Probability'] = prediction_probas
        results_df['Prediction'] = predictions
        results_df['Is_Fraud_Detected'] = prediction_probas > fraud_threshold
        results_df['Confidence_Level'] = pd.cut(prediction_probas, 
                                               bins=[0, 0.5, 0.8, 1.0], 
                                               labels=['LOW', 'MEDIUM', 'HIGH'])
        
        # Summary statistics
        fraud_detected_count = (prediction_probas > fraud_threshold).sum()
        high_confidence_fraud = (prediction_probas > 0.8).sum()
        
        print(f"Transactions flagged as fraud: {fraud_detected_count}")
        print(f"High confidence fraud detections: {high_confidence_fraud}")
        print(f"Average fraud probability: {prediction_probas.mean():.4f}")
        
        return results_df
        
    except Exception as e:
        print(f"❌ Error in batch prediction: {str(e)}")
        return None

def create_sample_transaction():
    """
    Create a sample transaction for testing
    """
    
    np.random.seed(42)
    
    # Create sample transaction with typical creditcard.csv features
    sample_transaction = {}
    
    # Add Transaction ID
    sample_transaction['Transaction_ID'] = 'TXN_' + str(np.random.randint(100000, 999999))
    
    # Time (in seconds from start)
    sample_transaction['Time'] = np.random.randint(0, 172800)  # 2 days
    
    # V features (PCA transformed features)
    for i in range(1, 29):
        sample_transaction[f'V{i}'] = np.random.normal(0, 1)
    
    # Amount
    sample_transaction['Amount'] = np.random.lognormal(3, 1.5)
    
    # Make some transactions more suspicious
    if np.random.random() < 0.1:  # 10% chance of suspicious features
        sample_transaction['Amount'] = np.random.uniform(1000, 5000)  # Higher amount
        sample_transaction['V1'] = np.random.normal(2, 1)  # Unusual V1
        sample_transaction['V2'] = np.random.normal(-2, 1)  # Unusual V2
        print("⚠️ Generated suspicious transaction for testing")
    
    return sample_transaction

def main():
    """
    Main function demonstrating model saving and prediction
    """
    
    print("=== MODEL SAVING & PREDICTION DEMO ===\n")
    
    # Step 1: Create a sample model for demonstration
    # (In practice, this would be your trained model from the previous step)
    
    try:
        # Try to create a simple model for demonstration
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        import xgboost as xgb
        
        print("1. Creating sample model for demonstration...")
        
        # Generate sample training data
        np.random.seed(42)
        n_samples = 1000
        n_features = 30
        
        X_sample = np.random.randn(n_samples, n_features)
        y_sample = np.random.choice([0, 1], n_samples, p=[0.99, 0.01])
        
        # Train a simple XGBoost model
        model = xgb.XGBClassifier(
            n_estimators=50,
            max_depth=4,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sample)
        
        model.fit(X_scaled, y_sample)
        print("Sample model trained successfully")
        
    except ImportError:
        print("❌ XGBoost not available, using RandomForest instead")
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sample)
        model.fit(X_scaled, y_sample)
    
    # Step 2: Save the model and scaler
    print("\n2. Saving model and scaler...")
    save_info = save_model_and_scaler(model, scaler, "xgboost_fraud_model")
    
    if save_info is None:
        print("❌ Failed to save model. Exiting...")
        return
    
    # Step 3: Load the model and scaler
    print("\n3. Loading saved model and scaler...")
    loaded_model, loaded_scaler = load_model_and_scaler(
        save_info['model_path'], 
        save_info['scaler_path']
    )
    
    if loaded_model is None:
        print("❌ Failed to load model. Exiting...")
        return
    
    # Step 4: Create sample transaction and predict
    print("\n4. Creating sample transaction for prediction...")
    sample_transaction = create_sample_transaction()
    
    # Convert to DataFrame format expected by creditcard.csv
    transaction_df = pd.DataFrame([sample_transaction])
    
    print("\nSample Transaction Details:")
    print(f"Transaction ID: {sample_transaction['Transaction_ID']}")
    print(f"Amount: ${sample_transaction['Amount']:.2f}")
    print(f"Time: {sample_transaction['Time']}")
    
    # Step 5: Make prediction
    print("\n5. Making fraud prediction...")
    
    # For demonstration, we'll create features that match our model
    # (In practice, your transaction would have the exact features your model expects)
    feature_data = {}
    for i in range(n_features):
        if i < 28:
            feature_data[f'V{i+1}'] = sample_transaction.get(f'V{i+1}', np.random.randn())
        elif i == 28:
            feature_data['Time'] = sample_transaction['Time']
        else:
            feature_data['Amount'] = sample_transaction['Amount']
    
    feature_data['Transaction_ID'] = sample_transaction['Transaction_ID']
    
    # Make prediction with threshold of 0.7
    result = predict_single_transaction(
        loaded_model, 
        loaded_scaler, 
        feature_data, 
        fraud_threshold=0.7
    )
    
    # Step 6: Display final result
    if result:
        print("\n" + "="*60)
        print("FINAL PREDICTION SUMMARY")
        print("="*60)
        print(f"Transaction ID: {result['transaction_id']}")
        print(f"Amount: ${result['amount']:.2f}")
        print(f"Fraud Probability: {result['fraud_probability']:.4f}")
        print(f"Confidence Level: {result['confidence_level']}")
        print(f"Fraud Detected: {'YES' if result['is_fraud_detected'] else 'NO'}")
        
        if result['is_fraud_detected']:
            print("\n🚨 FRAUD DETECTED! 🚨")
            print("Alert systems should be triggered!")
    
    # Step 7: Demonstrate batch prediction
    print("\n6. Demonstrating batch prediction...")
    
    # Create multiple sample transactions
    batch_transactions = []
    for i in range(5):
        sample_txn = create_sample_transaction()
        # Convert to model features format
        txn_features = {}
        for j in range(n_features):
            if j < 28:
                txn_features[f'V{j+1}'] = sample_txn.get(f'V{j+1}', np.random.randn())
            elif j == 28:
                txn_features['Time'] = sample_txn['Time']
            else:
                txn_features['Amount'] = sample_txn['Amount']
        txn_features['Transaction_ID'] = sample_txn['Transaction_ID']
        batch_transactions.append(txn_features)
    
    batch_df = pd.DataFrame(batch_transactions)
    batch_results = predict_batch_transactions(loaded_model, loaded_scaler, batch_df)
    
    if batch_results is not None:
        print("\nBatch Prediction Results:")
        fraud_detected = batch_results[batch_results['Is_Fraud_Detected']]
        if len(fraud_detected) > 0:
            print(f"Fraud detected in {len(fraud_detected)} transactions:")
            for _, row in fraud_detected.iterrows():
                print(f"  - {row['Transaction_ID']}: ${row['Amount']:.2f} (Probability: {row['Fraud_Probability']:.4f})")
        else:
            print("No fraud detected in batch transactions.")
    
    print("\n=== MODEL SAVING & PREDICTION DEMO COMPLETE ===")
    
    return {
        'model': loaded_model,
        'scaler': loaded_scaler,
        'save_info': save_info,
        'sample_prediction': result
    }

if __name__ == "__main__":
    demo_results = main()