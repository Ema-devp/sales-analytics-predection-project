import pandas as pd
import numpy as np

def load_and_preprocess_data(filepath):
    # Load dataset
    df = pd.read_csv(filepath, parse_dates=['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Time-based features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Lag features (previous sales to help predict future sales)
    df['sales_lag_1'] = df['sales'].shift(1)
    df['sales_lag_7'] = df['sales'].shift(7)
    
    # Rolling window statistics
    df['sales_rolling_mean_7'] = df['sales'].shift(1).rolling(window=7).mean()
    
    # Drop rows with NaN values created by lags/rolling windows
    df = df.dropna().reset_index(drop=True)
    
    return df