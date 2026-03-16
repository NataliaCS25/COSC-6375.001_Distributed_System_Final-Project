import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

BASE_PATH = "/mnt/c/Users/natal/OneDrive/Documents/Spring2026/Project-copy/ERCOT_Federated_Forecasting"
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
    df = df.dropna().reset_index(drop=True) # drop rows with NaNs from shifting

    s_target = MinMaxScaler()
    s_feat = MinMaxScaler()

    # target is actual zonal load
    t_scaled = s_target.fit_transform(df[[ZONE_MAP[zone_name]]].values)
    # Scale lags using the target scaler
    lag1_scaled = s_target.transform(df[['Lag_1']].values)
    lag24_scaled = s_target.transform(df[['Lag_24']].values)
    # Weather
    f_scaled = s_feat.fit_transform(df[['Temperature Average']].values)
    # Temporal features

    # df = pd.merge(df_load[['time_utc', ZONE_MAP[zone_name], 'Demand forecast']], df_w, on='time_utc').dropna()
    # df = df[df['time_utc'].dt.year == 2023].reset_index(drop=True)
    # df['residual'] = df[ZONE_MAP[zone_name]] - df['Demand forecast']
    
    # Fit local scalers per zone to maximize variance and reveal the daily shape
    # s_target = MinMaxScaler()
    # s_feat = MinMaxScaler()

    # Scaling (using .values to silence warnings)
    # t_scaled = s_target.fit_transform(df[['residual']].values)
    # f_scaled = s_feat.fit_transform(df[['Temperature Average']].values)
    
    # Temporal Features (4 columns)
    h_sin = np.sin(2 * np.pi * df['time_utc'].dt.hour / 24).values.reshape(-1, 1)
    h_cos = np.cos(2 * np.pi * df['time_utc'].dt.hour / 24).values.reshape(-1, 1)
    d_sin = np.sin(2 * np.pi * df['time_utc'].dt.dayofweek / 7).values.reshape(-1, 1)
    d_cos = np.cos(2 * np.pi * df['time_utc'].dt.dayofweek / 7).values.reshape(-1, 1)
    
  
Regional One-Hot (3 columns)
    reg_oh = np.zeros((len(df), len(ZONES)))
    reg_oh[:, ZONES.index(zone_name)] = 1
    
    # TOTAL FEATURES = 2 (lags) + 1 (temp) + 4 (time) + 3 (zones) = 10 features
    full_data = np.hstack([lag1_scaled, lag24_scaled, f_scaled, h_sin, h_cos, d_sin, d_cos, reg_oh])
    
    X, y, timestamps, baseline_list = [], [], [], []
    for i in range(len(full_data) - (seq_length+ 24)):
        X.append(full_data[i : i + seq_length])

        # Target window (the next 24h)
        y.append(t_scaled[i + seq_length : i + seq_length + 24].flatten()) # target is the actual zonal load at t+1, not the residual

        window_ts = df['time_utc'].iloc[i + seq_length : i + seq_length + 24].values
        timestamps.append(window_ts) # Capture timestamps for the target window

        window_baseline = df['Demand forecast'].iloc[i + seq_length : i + seq_length + 24].values
        baseline_list.append(window_baseline) # Capture baseline for the target window
      
    return np.array(X), np.array(y), s_target, timestamps, baseline_list # baseline is not needed for training, only for plotting later

def get_dataloader(X, y, batch_size=64, shuffle=True):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), 
                            torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
