import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           precision_score, recall_score, f1_score, 
                           roc_auc_score, roc_curve, precision_recall_curve)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def calculate_class_weights(y):
    """Calculate class weights for handling imbalanced dataset"""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    class_weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, class_weights))
    
    print(f"Class weights: {class_weight_dict}")
    
    # For XGBoost scale_pos_weight calculation
    scale_pos_weight = (y == 0).sum() / (y == 1).sum()
    print(f"XGBoost scale_pos_weight: {scale_pos_weight:.2f}")
    
    return class_weight_dict, scale_pos_weight

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest with class balancing"""
    
    print("=== TRAINING RANDOM FOREST CLASSIFIER ===")
    
    # Calculate class weights
    class_weights, _ = calculate_class_weights(y_train)
    
    # Initialize and train Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',  # Handle class imbalance
        random_state=42,
        n_jobs=-1
    )
    
    print("Training Random Forest...")
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    
    print("Random Forest training completed!")
    
    return rf_model, rf_pred, rf_pred_proba

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost with class balancing"""
    
    print("\n=== TRAINING XGBOOST CLASSIFIER ===")
    
    # Calculate scale_pos_weight for XGBoost
    _, scale_pos_weight = calculate_class_weights(y_train)
    
    # Initialize and train XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    print("Training XGBoost...")
    xgb_model.fit(X_train, y_train)
    
    # Make predictions
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    
    print("XGBoost training completed!")
    
    return xgb_model, xgb_pred, xgb_pred_proba

def evaluate_model(y_true, y_pred, y_pred_proba, model_name):
    """Comprehensive model evaluation"""
    
    print(f"\n=== {model_name.upper()} EVALUATION ===")
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred_proba)
    
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC: {roc_auc:.4f}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': cm
    }

def plot_model_comparison(y_test, rf_pred_proba, xgb_pred_proba, rf_metrics, xgb_metrics):
    """Plot model comparison visualizations"""
    
    plt.figure(figsize=(15, 10))
    
    # ROC Curves
    plt.subplot(2, 3, 1)
    
    # Random Forest ROC
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_pred_proba)
    plt.plot(rf_fpr, rf_tpr, label=f'Random Forest (AUC = {rf_metrics["roc_auc"]:.4f})', linewidth=2)
    
    # XGBoost ROC
    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_pred_proba)
    plt.plot(xgb_fpr, xgb_tpr, label=f'XGBoost (AUC = {xgb_metrics["roc_auc"]:.4f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Precision-Recall Curves
    plt.subplot(2, 3, 2)
    
    # Random Forest PR
    rf_precision, rf_recall, _ = precision_recall_curve(y_test, rf_pred_proba)
    plt.plot(rf_recall, rf_precision, label=f'Random Forest', linewidth=2)
    
    # XGBoost PR
    xgb_precision, xgb_recall, _ = precision_recall_curve(y_test, xgb_pred_proba)
    plt.plot(xgb_recall, xgb_precision, label=f'XGBoost', linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Metrics Comparison Bar Chart
    plt.subplot(2, 3, 3)
    metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    rf_values = [rf_metrics['precision'], rf_metrics['recall'], rf_metrics['f1'], rf_metrics['roc_auc']]
    xgb_values = [xgb_metrics['precision'], xgb_metrics['recall'], xgb_metrics['f1'], xgb_metrics['roc_auc']]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    plt.bar(x - width/2, rf_values, width, label='Random Forest', alpha=0.8)
    plt.bar(x + width/2, xgb_values, width, label='XGBoost', alpha=0.8)
    
    plt.xlabel('Metrics')
    plt.ylabel('Score')
    plt.title('Model Metrics Comparison')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confusion Matrix - Random Forest
    plt.subplot(2, 3, 4)
    sns.heatmap(rf_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Confusion Matrix - XGBoost
    plt.subplot(2, 3, 5)
    sns.heatmap(xgb_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title('XGBoost - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Prediction Probability Distribution
    plt.subplot(2, 3, 6)
    
    # Separate probabilities for each class
    normal_probs_rf = rf_pred_proba[y_test == 0]
    fraud_probs_rf = rf_pred_proba[y_test == 1]
    normal_probs_xgb = xgb_pred_proba[y_test == 0]
    fraud_probs_xgb = xgb_pred_proba[y_test == 1]
    
    plt.hist(normal_probs_rf, bins=50, alpha=0.5, label='RF - Normal', density=True)
    plt.hist(fraud_probs_rf, bins=50, alpha=0.5, label='RF - Fraud', density=True)
    plt.hist(normal_probs_xgb, bins=50, alpha=0.5, label='XGB - Normal', density=True)
    plt.hist(fraud_probs_xgb, bins=50, alpha=0.5, label='XGB - Fraud', density=True)
    
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Prediction Probability Distribution')
    plt.legend()
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

def cross_validate_models(X, y, cv_folds=5):
    """Perform cross-validation for both models"""
    
    print(f"\n=== CROSS-VALIDATION ({cv_folds}-FOLD) ===")
    
    # Stratified K-Fold for imbalanced dataset
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Random Forest
    _, scale_pos_weight = calculate_class_weights(y)
    
    rf_model = RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight='balanced', 
        random_state=42, n_jobs=-1
    )
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, scale_pos_weight=scale_pos_weight,
        random_state=42, eval_metric='logloss', use_label_encoder=False
    )
    
    # Cross-validation scores
    rf_scores = cross_val_score(rf_model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    xgb_scores = cross_val_score(xgb_model, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
    
    print(f"Random Forest CV ROC-AUC: {rf_scores.mean():.4f} (+/- {rf_scores.std() * 2:.4f})")
    print(f"XGBoost CV ROC-AUC: {xgb_scores.mean():.4f} (+/- {xgb_scores.std() * 2:.4f})")
    
    return rf_scores, xgb_scores

def main():
    """Main function to execute model training pipeline"""
    
    try:
        # Load dataset
        print("Loading dataset...")
        df = pd.read_csv('creditcard.csv')
        print(f"Dataset loaded with shape: {df.shape}")
        
    except FileNotFoundError:
        print("creditcard.csv not found. Creating sample dataset for demonstration...")
        np.random.seed(42)
        n_samples = 10000
        
        # Create sample dataset with similar structure
        df = pd.DataFrame()
        
        # Add V features (PCA components)
        for i in range(1, 29):
            df[f'V{i}'] = np.random.normal(0, 1, n_samples)
        
        df['Time'] = np.sort(np.random.randint(0, 172800, n_samples))
        df['Amount'] = np.random.lognormal(3, 1.5, n_samples)
        
        # Create imbalanced target (0.2% fraud)
        df['Class'] = np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
        
        print(f"Sample dataset created with shape: {df.shape}")
    
    # Prepare data
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print(f"\nClass distribution:")
    print(y.value_counts())
    print(f"Fraud percentage: {(y.sum() / len(y)) * 100:.4f}%")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Features scaled using StandardScaler")
    
    # Train models
    rf_model, rf_pred, rf_pred_proba = train_random_forest(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    xgb_model, xgb_pred, xgb_pred_proba = train_xgboost(
        X_train_scaled, y_train, X_test_scaled, y_test
    )
    
    # Evaluate models
    rf_metrics = evaluate_model(y_test, rf_pred, rf_pred_proba, "Random Forest")
    xgb_metrics = evaluate_model(y_test, xgb_pred, xgb_pred_proba, "XGBoost")
    
    # Cross-validation
    rf_cv_scores, xgb_cv_scores = cross_validate_models(X_train_scaled, y_train)
    
    # Plot comparisons
    plot_model_comparison(y_test, rf_pred_proba, xgb_pred_proba, rf_metrics, xgb_metrics)
    
    # Model comparison summary
    print("\n" + "="*50)
    print("FINAL MODEL COMPARISON SUMMARY")
    print("="*50)
    
    comparison_df = pd.DataFrame({
        'Random Forest': [rf_metrics['precision'], rf_metrics['recall'], 
                         rf_metrics['f1'], rf_metrics['roc_auc']],
        'XGBoost': [xgb_metrics['precision'], xgb_metrics['recall'], 
                   xgb_metrics['f1'], xgb_metrics['roc_auc']]
    }, index=['Precision', 'Recall', 'F1-Score', 'ROC-AUC'])
    
    print(comparison_df.round(4))
    
    # Determine best model
    if xgb_metrics['roc_auc'] > rf_metrics['roc_auc']:
        best_model = xgb_model
        best_model_name = "XGBoost"
        print(f"\nBest model: {best_model_name} (ROC-AUC: {xgb_metrics['roc_auc']:.4f})")
    else:
        best_model = rf_model
        best_model_name = "Random Forest"
        print(f"\nBest model: {best_model_name} (ROC-AUC: {rf_metrics['roc_auc']:.4f})")
    
    print("\n=== TRAINING COMPLETE ===")
    
    return {
        'rf_model': rf_model,
        'xgb_model': xgb_model,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'scaler': scaler,
        'rf_metrics': rf_metrics,
        'xgb_metrics': xgb_metrics
    }

if __name__ == "__main__":
    results = main()
    