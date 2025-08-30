import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def engineer_features(df):
    """
    Engineer features for fraud detection:
    1. Transaction frequency per user
    2. Average transaction amount per user  
    3. Time since last transaction
    """
    
    # Create a copy to avoid modifying original data
    df_engineered = df.copy()
    
    # For this dataset, we'll create a synthetic user ID since it's not provided
    # In real scenarios, you would have actual user IDs
    print("Creating synthetic user IDs for demonstration...")
    
    # Create user groups based on patterns in the data
    # This is a simplified approach - in reality you'd have actual user IDs
    np.random.seed(42)
    n_users = len(df) // 50  # Assume each user has ~50 transactions on average
    df_engineered['User_ID'] = np.random.randint(1, n_users + 1, size=len(df))
    
    print(f"Created {n_users} unique users")
    print(f"Average transactions per user: {len(df) / n_users:.2f}")
    
    # Sort by User_ID and Time for proper time-based calculations
    df_engineered = df_engineered.sort_values(['User_ID', 'Time'])
    
    # Feature 1: Transaction frequency per user
    print("\n1. Engineering transaction frequency per user...")
    user_freq = df_engineered.groupby('User_ID').size()
    df_engineered['Transaction_Frequency'] = df_engineered['User_ID'].map(user_freq)
    
    # Feature 2: Average transaction amount per user
    print("2. Engineering average transaction amount per user...")
    user_avg_amount = df_engineered.groupby('User_ID')['Amount'].mean()
    df_engineered['User_Avg_Amount'] = df_engineered['User_ID'].map(user_avg_amount)
    
    # Feature 3: Time since last transaction
    print("3. Engineering time since last transaction...")
    df_engineered['Time_Since_Last_Transaction'] = df_engineered.groupby('User_ID')['Time'].diff()
    
    # Fill NaN values (first transaction for each user) with median
    median_time_diff = df_engineered['Time_Since_Last_Transaction'].median()
    df_engineered['Time_Since_Last_Transaction'].fillna(median_time_diff, inplace=True)
    
    # Additional derived features
    print("4. Engineering additional derived features...")
    
    # Amount deviation from user's average
    df_engineered['Amount_Deviation_From_User_Avg'] = (
        df_engineered['Amount'] - df_engineered['User_Avg_Amount']
    ) / (df_engineered['User_Avg_Amount'] + 1e-8)  # Add small value to avoid division by zero
    
    # Transaction velocity (frequency / time range)
    user_time_range = df_engineered.groupby('User_ID')['Time'].apply(lambda x: x.max() - x.min() + 1)
    df_engineered['User_Time_Range'] = df_engineered['User_ID'].map(user_time_range)
    df_engineered['Transaction_Velocity'] = df_engineered['Transaction_Frequency'] / df_engineered['User_Time_Range']
    
    # Hour of day (assuming Time is seconds from start)
    df_engineered['Hour_of_Day'] = (df_engineered['Time'] % 86400) // 3600  # 86400 seconds in a day
    
    # Day of week (simplified)
    df_engineered['Day_of_Week'] = (df_engineered['Time'] // 86400) % 7
    
    # Amount percentile within user's transactions
    df_engineered['Amount_Percentile_User'] = df_engineered.groupby('User_ID')['Amount'].rank(pct=True)
    
    print("\n=== FEATURE ENGINEERING SUMMARY ===")
    print("New features created:")
    new_features = [
        'User_ID', 'Transaction_Frequency', 'User_Avg_Amount', 
        'Time_Since_Last_Transaction', 'Amount_Deviation_From_User_Avg',
        'User_Time_Range', 'Transaction_Velocity', 'Hour_of_Day', 
        'Day_of_Week', 'Amount_Percentile_User'
    ]
    
    for feature in new_features:
        print(f"- {feature}")
    
    print(f"\nOriginal features: {df.shape[1]}")
    print(f"Total features after engineering: {df_engineered.shape[1]}")
    print(f"New features added: {len(new_features)}")
    
    # Display statistics for new features
    print("\n=== NEW FEATURES STATISTICS ===")
    for feature in ['Transaction_Frequency', 'User_Avg_Amount', 'Time_Since_Last_Transaction']:
        print(f"\n{feature}:")
        print(df_engineered[feature].describe())
    
    # Check for any infinite or NaN values
    print("\n=== DATA QUALITY CHECK ===")
    inf_count = np.isinf(df_engineered.select_dtypes(include=[np.number])).sum().sum()
    nan_count = df_engineered.isnull().sum().sum()
    print(f"Infinite values: {inf_count}")
    print(f"NaN values: {nan_count}")
    
    if inf_count > 0:
        print("Replacing infinite values with large finite values...")
        df_engineered = df_engineered.replace([np.inf, -np.inf], [1e10, -1e10])
    
    return df_engineered

def prepare_features_for_training(df_engineered, target_col='Class'):
    """
    Prepare the engineered features for model training
    """
    
    # Separate features and target
    feature_cols = [col for col in df_engineered.columns if col != target_col]
    X = df_engineered[feature_cols]
    y = df_engineered[target_col]
    
    print(f"\n=== TRAINING DATA PREPARATION ===")
    print(f"Feature columns: {len(feature_cols)}")
    print(f"Sample size: {len(X)}")
    print(f"Target distribution:")
    print(y.value_counts())
    
    return X, y

# Example usage with correlation analysis
def analyze_new_features(df_engineered):
    """
    Analyze the relationship between new features and fraud
    """
    
    print("\n=== FEATURE CORRELATION ANALYSIS ===")
    
    # Calculate correlation with target variable
    new_features = [
        'Transaction_Frequency', 'User_Avg_Amount', 'Time_Since_Last_Transaction',
        'Amount_Deviation_From_User_Avg', 'Transaction_Velocity', 
        'Hour_of_Day', 'Day_of_Week', 'Amount_Percentile_User'
    ]
    
    correlations = df_engineered[new_features + ['Class']].corr()['Class'].sort_values(key=abs, ascending=False)
    
    print("Correlation with fraud (Class):")
    for feature in correlations.index[:-1]:  # Exclude 'Class' itself
        print(f"{feature}: {correlations[feature]:.4f}")
    
    # Compare feature values between fraud and non-fraud
    print("\n=== FEATURE COMPARISON: FRAUD vs NON-FRAUD ===")
    for feature in new_features[:4]:  # Show first 4 features
        fraud_mean = df_engineered[df_engineered['Class'] == 1][feature].mean()
        normal_mean = df_engineered[df_engineered['Class'] == 0][feature].mean()
        print(f"{feature}:")
        print(f"  Normal transactions: {normal_mean:.4f}")
        print(f"  Fraud transactions: {fraud_mean:.4f}")
        print(f"  Difference ratio: {fraud_mean/normal_mean:.4f}" if normal_mean != 0 else "  Difference ratio: inf")
        print()

# Main execution function
def main():
    """
    Main function to demonstrate feature engineering
    """
    
    # Load sample data (in practice, this would be your loaded dataset)
    try:
        df = pd.read_csv('creditcard.csv')
        print(f"Loaded dataset with shape: {df.shape}")
    except FileNotFoundError:
        print("Creating sample data for demonstration...")
        # Create sample data structure
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'Time': np.sort(np.random.randint(0, 172800, n_samples)),  # 2 days in seconds
            'Amount': np.random.lognormal(3, 1.5, n_samples),
            'Class': np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
        })
        # Add some V features (like in the real dataset)
        for i in range(1, 11):
            df[f'V{i}'] = np.random.normal(0, 1, n_samples)
        
        print(f"Created sample dataset with shape: {df.shape}")
    
    # Engineer features
    df_engineered = engineer_features(df)
    
    # Analyze new features
    analyze_new_features(df_engineered)
    
    # Prepare for training
    X, y = prepare_features_for_training(df_engineered)
    
    print("\n=== FEATURE ENGINEERING COMPLETE ===")
    print("Enhanced dataset ready for model training!")
    
    return df_engineered, X, y

if __name__ == "__main__":
    df_engineered, X, y = main()