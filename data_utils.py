import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LinearRegression
import statsmodels.tsa.arima.model

BASE_PATH = "."
ZONES = ["NCENT", "COAST", "FWEST"] # Exactly 3 zones = 3 one-hot columns
ZONE_MAP = {"NCENT": "Subregion NCEN", "COAST": "Subregion COAS", "FWEST": "Subregion FWES"}


def prepare_zonal_data(zone_name, load_path, weather_path, seq_length=24):
    df_load = pd.read_excel(load_path)
    df_w = pd.read_excel(weather_path)
    df_load['time_utc'] = pd.to_datetime(df_load['UTC time'], utc=True).dt.tz_localize(None)
    df_w['time_utc'] = pd.to_datetime(df_w['Datetime ISO8601 (UTC)'], utc=True).dt.tz_localize(None)

    df = pd.merge(df_load[['time_utc', ZONE_MAP[zone_name], 'Demand forecast']], df_w, on='time_utc').dropna()
    df = df.sort_values('time_utc').reset_index(drop=True)

    # Implement lag features
    df['Lag_1'] = df[ZONE_MAP[zone_name]].shift(1)
    df['Lag_24'] = df[ZONE_MAP[zone_name]].shift(24)
    df['Lag_168'] = df[ZONE_MAP[zone_name]].shift(168) # 1 week lag
    df['Rolling_Mean_24h'] = df['Lag_1'].rolling(window=24).mean() # Rolling mean of the last 24h
    df = df.dropna().reset_index(drop=True) # drop rows with NaNs from shifting

    train_size = int(len(df)*0.8) # Fit scaler only on the first 80% of the data to prevent data leakage

    s_target = MinMaxScaler()
    s_feat = MinMaxScaler()

    # Fit ONLY on train data
    s_target.fit(df[[ZONE_MAP[zone_name]]].iloc[:train_size].values)
    s_feat.fit(df[['Temperature Average']].iloc[:train_size].values)

    # Transform ALL data
    t_scaled = s_target.transform(df[[ZONE_MAP[zone_name]]].values)
    # Scale lags using the target scaler
    lag1_scaled = s_target.transform(df[['Lag_1']].values)
    lag24_scaled = s_target.transform(df[['Lag_24']].values)
    lag168_scaled = s_target.transform(df[['Lag_168']].values)
    roll_mean_scaled = s_target.transform(df[['Rolling_Mean_24h']].values)
    # Weather
    f_scaled = s_feat.transform(df[['Temperature Average']].values)
    
    # 2. Temporal Features (4 columns)
    h_sin = np.sin(2 * np.pi * df['time_utc'].dt.hour / 24).values.reshape(-1, 1)
    h_cos = np.cos(2 * np.pi * df['time_utc'].dt.hour / 24).values.reshape(-1, 1)
    d_sin = np.sin(2 * np.pi * df['time_utc'].dt.dayofweek / 7).values.reshape(-1, 1)
    d_cos = np.cos(2 * np.pi * df['time_utc'].dt.dayofweek / 7).values.reshape(-1, 1)
    
    # 3. Regional One-Hot (3 columns)
    reg_oh = np.zeros((len(df), len(ZONES)))
    reg_oh[:, ZONES.index(zone_name)] = 1
    
    # TOTAL FEATURES = 4 (lags) + 1 (temp) + 4 (time) + 3 (zones) = 12 features
    full_data = np.hstack([lag1_scaled, lag24_scaled, lag168_scaled, roll_mean_scaled, f_scaled, h_sin, h_cos, d_sin, d_cos, reg_oh])
    

    # [RAI UPDATE] Added temp_list to store raw temperatures for fairness slicing
    X, y, timestamps, baseline_list, temp_list = [], [], [], [], []
    #X, y, timestamps, baseline_list = [], [], [], []
    for i in range(len(full_data) - (seq_length+ 24)):
        X.append(full_data[i : i + seq_length])

        # Target window (the next 24h)
        y.append(t_scaled[i + seq_length : i + seq_length + 24].flatten()) # target is the actual zonal load at t+1, not the residual

        window_ts = df['time_utc'].iloc[i + seq_length : i + seq_length + 24].values
        timestamps.append(window_ts) # Capture timestamps for the target window

        window_baseline = df['Demand forecast'].iloc[i + seq_length : i + seq_length + 24].values
        baseline_list.append(window_baseline) # Capture baseline for the target window

        # Capture raw temperature for the target window
        window_temp = df['Temperature Average'].iloc[i + seq_length : i + seq_length + 24].values
        temp_list.append(window_temp)
    
    # RETURN 5 VALUES to match main.py line 25
    return np.array(X), np.array(y), s_target, timestamps, baseline_list, np.array(temp_list) 

def get_dataloader(X, y, batch_size=64, shuffle=True):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), 
                            torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def get_persistence_baseline(actual_values, horizon=24):
    """
    Predicts the next 24 hours by using the previous 24 hours.
    actual_values: The historical load series (MW)
    """
    # Simply shift the data by the horizon
    persistence_pred = actual_values[-horizon:] 
    return persistence_pred


def train_linear_baseline(X_train, y_train, X_test):
    """
    X_train: (N, features) - Flattened sequence if needed
    y_train: (N, 24) - The 24h target window
    """
    # Linear Regression requires 2D input (flattening the time steps)
    N, seq_len, num_feats = X_train.shape
    X_train_flat = X_train.reshape(N, -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)
    
    model = LinearRegression()
    model.fit(X_train_flat, y_train)
    
    preds = model.predict(X_test_flat)
    return preds


def get_arima_baseline(history, horizon=24):
    """
    history: List of past MW values for the zone
    """
    # Order (p,d,q) - (5,1,0) is a standard starting point for hourly load
    model = statsmodels.tsa.arima.model.ARIMA(history, order=(5, 1, 0))
    model_fit = model.fit()
    
    # Forecast the next 24 hours
    forecast = model_fit.forecast(steps=horizon)
    return forecast
