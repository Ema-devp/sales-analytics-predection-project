import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from preprocessing import load_and_preprocess_data

def train_model(data_path):
    df = load_and_preprocess_data(data_path)
    
    # Define features and target
    features = [
        'month', 'day', 'day_of_week', 'is_weekend', 
        'sales_lag_1', 'sales_lag_7', 'sales_rolling_mean_7'
    ]
    target = 'sales'
    
    X = df[features]
    y = df[target]
    
    # Time-series split (keeping chronological order)
    train_size = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    
    # Initialize and train model
    model = XGBRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Model Training Complete. Test MAE: {mae:.2f}")
    
    # Save model artifact
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/sales_model.pkl')
    print("Model saved to models/sales_model.pkl")

if __name__ == "__main__":
    # Example usage: point to your raw sales CSV file
    # train_model('data/raw_sales.csv')