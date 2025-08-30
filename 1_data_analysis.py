import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Load the dataset
def load_data(file_path='creditcard.csv'):
    """Load the credit card fraud dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"File {file_path} not found. Please ensure the file exists.")
        return None

# Basic EDA
def perform_eda(df):
    """Perform basic exploratory data analysis"""
    print("=== BASIC DATASET INFO ===")
    print(f"Dataset shape: {df.shape}")
    print(f"\nColumn names: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    
    print("\n=== NULL VALUE CHECK ===")
    null_counts = df.isnull().sum()
    print(f"Null values per column:\n{null_counts}")
    print(f"Total null values: {null_counts.sum()}")
    
    print("\n=== FRAUD VS NON-FRAUD DISTRIBUTION ===")
    fraud_counts = df['Class'].value_counts()
    print(f"Class distribution:\n{fraud_counts}")
    print(f"Fraud percentage: {(fraud_counts[1] / len(df)) * 100:.4f}%")
    
    # Visualizations
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    df['Class'].value_counts().plot(kind='bar')
    plt.title('Class Distribution')
    plt.xlabel('Class (0: Normal, 1: Fraud)')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    plt.subplot(1, 2, 2)
    plt.hist(df[df['Class'] == 0]['Amount'], bins=50, alpha=0.7, label='Normal', density=True)
    plt.hist(df[df['Class'] == 1]['Amount'], bins=50, alpha=0.7, label='Fraud', density=True)
    plt.xlabel('Transaction Amount')
    plt.ylabel('Density')
    plt.title('Transaction Amount Distribution')
    plt.legend()
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()
    
    print("\n=== BASIC STATISTICS ===")
    print(f"Amount statistics for Normal transactions:")
    print(df[df['Class'] == 0]['Amount'].describe())
    print(f"\nAmount statistics for Fraud transactions:")
    print(df[df['Class'] == 1]['Amount'].describe())

# Feature scaling and data splitting
def preprocess_and_split(df, test_size=0.2, random_state=42):
    """Scale features and split data with stratified sampling"""
    
    # Separate features and target
    X = df.drop(['Class'], axis=1)
    y = df['Class']
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # Split the data with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Training set fraud percentage: {(y_train.sum() / len(y_train)) * 100:.4f}%")
    print(f"Test set fraud percentage: {(y_test.sum() / len(y_test)) * 100:.4f}%")
    
    # Scale the features
    scaler = StandardScaler()
    
    # Fit scaler on training data and transform both train and test
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames to maintain column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)
    
    print("\n=== SCALING COMPLETED ===")
    print("Features have been standardized using StandardScaler")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# Main execution
def main():
    """Main function to execute data preprocessing pipeline"""
    
    # Load data
    df = load_data('creditcard.csv')
    if df is None:
        return
    
    # Perform EDA
    perform_eda(df)
    
    # Preprocess and split data
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(df)
    
    print("\n=== PREPROCESSING COMPLETE ===")
    print("Data is ready for model training!")
    
    # Return processed data for further use
    return X_train, X_test, y_train, y_test, scaler

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, scaler = main()