import flwr as fl
import shap
import torch
import torch.nn as nn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg') 
import numpy as np
from flwr.common import parameters_to_ndarrays
from sklearn.metrics import mean_squared_error, mean_absolute_error # RAI UPDATE
from data_utils import prepare_zonal_data, get_dataloader, ZONES, BASE_PATH, get_persistence_baseline, train_linear_baseline, get_arima_baseline
from model import BiLSTMForecaster, StandardLSTMForecaster, LinearForecaster
from client import ERCOTClient
import logging
import os
import pickle
import warnings
import json
# Ignores the specific DeprecationWarning from Flower
warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr")

# Silence the "RpcError" and metrics exporter noise
os.environ["RAY_metrics_export_binaries_path"] = ""
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

LOAD_FILE = f"{BASE_PATH}/Demand_Data.xlsx"

def aggregate_metrics(metrics):
    if not metrics: return {}
    total = sum([n for n, _ in metrics])
    return {"rmse": sum([n * m["rmse"] for n, m in metrics]) / total,
            "mape": sum([n * m["mape"] for n, m in metrics]) / total,
            "nmbe": sum([n * m["nmbe"] for n, m in metrics]) / total}

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, model_class, save_name, **kwargs):
        super().__init__(**kwargs)
        self.model_class = model_class
        self.save_name = save_name

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            model = self.model_class(input_dim=12) # Instantiate the specific model
            params_dict = zip(model.state_dict().keys(), ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            
            torch.save(model.state_dict(), f"{BASE_PATH}/{self.save_name}.pt")
            print(f"{self.save_name} saved at round {server_round}!")
            
        return aggregated_parameters, aggregated_metrics
    
# ==========================================
# [RAI UPDATE]
# ==========================================
def run_fairness_audit(y_true, y_pred, temps, model_name, zone):
    """Pillar 3: Fairness - Evaluates Temperature Fairness Gap"""
    y_true_f, y_pred_f, temps_f = y_true.flatten(), y_pred.flatten(), temps.flatten()
    
    low_idx = temps_f < 40
    high_idx = temps_f > 77
    
    mae_low = mean_absolute_error(y_true_f[low_idx], y_pred_f[low_idx]) if np.any(low_idx) else 0
    mae_high = mean_absolute_error(y_true_f[high_idx], y_pred_f[high_idx]) if np.any(high_idx) else 0
    
    gap = mae_low - mae_high
    print(f"[{zone}] {model_name} Fairness Gap (Low - High T): {gap:.4f}")
    return gap

def generate_model_card(model_name, rmse, mape, fairness_gaps):
    """Pillar 1: Governance - Generates the standardized Model Card"""
    card = {
        "Intended Use": "Short-Term Electricity Load Forecasting for grid-level demand planning",
        "Model Identification": {"Architecture": model_name},
        "Metrics": {"Global RMSE": float(rmse), "Global MAPE": float(mape)},
        "Explainability Summary": "SHAP integrated. Critical features: Temporal Lags & Temperature.",
        "Fairness Audit (Temperature Gap)": fairness_gaps,
        "Limitations": "Model reliability strictly bounded to weather distributions seen between 2020-2025."
    }
    clean_name = model_name.lower().replace(' ', '_')
    with open(f"{BASE_PATH}/{clean_name}_model_card.json", "w") as f:
        json.dump(card, f, indent=4)
    print(f"Model Card safely generated for {model_name}")

def run_shap_analysis(model, X_train_tensor, X_test_tensor, model_name):
    """Pillar 2: Explainability - Generates and saves SHAP plots"""
    print(f"--- Generating SHAP Plots for {model_name} ---")
    background = X_train_tensor[:100] 
    test_sample = X_test_tensor[:100]
    
    feature_names = ["Lag_1", "Lag_24", "Lag_168", "Roll_Mean_24", "Temp", "Hr_Sin", "Hr_Cos", "Day_Sin", "Day_Cos", "NCENT", "COAST", "FWEST"]
    
    try:
        explainer = shap.DeepExplainer(model, background)
        
        # SHAP values for the first prediction hour [0] to understand immediate feature importance.
        shap_values = explainer.shap_values(test_sample)[0] 
        
        # Because input is 3D (batch, 168, 12), we average across the 168-hour sequence length 
        # to get the overall global importance of each of the 12 features
        shap_values_2d = np.abs(shap_values).mean(axis=1)
        test_sample_2d = test_sample.numpy().mean(axis=1)

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values_2d, test_sample_2d, feature_names=feature_names, show=False)
        clean_name = model_name.lower().replace(' ', '_')
        plt.savefig(f"{BASE_PATH}/shap_summary_{clean_name}.png", bbox_inches='tight')
        plt.close()
        print(f"SHAP summary plot saved for {model_name}!")
    except Exception as e:
        print(f"SHAP calculation skipped: {e}")


# ==========================================
# THE THREE-WAY COMPARATIVE AUDIT
# ==========================================

def run_centralized(model_class, model_name, epochs):
    """NEW: Centralized Training (Data Lake) representing the upper-bound baseline"""
    print(f"\n--- Running Centralized Training: {model_name} ---")
    
    X_train_all, y_train_all = [], []
    X_test_dict, y_test_dict = {}, {}

    # 1. Pool all data together
    for zone in ZONES:
        X, y, _, _, _, _ = prepare_zonal_data(zone, LOAD_FILE, f"{BASE_PATH}/{zone}_TX_Total.xlsx")
        split = int(0.8 * len(X))
        
        X_train_all.append(X[:split])
        y_train_all.append(y[:split])
        
        X_test_dict[zone] = X[split:]
        y_test_dict[zone] = y[split:]

    X_train_central = np.concatenate(X_train_all, axis=0)
    y_train_central = np.concatenate(y_train_all, axis=0)
    
    train_loader = get_dataloader(X_train_central, y_train_central, shuffle=True)
    model = model_class(input_dim=12)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        batch_losses = []
        for bx, by in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        print(f"[Centralized] Epoch {epoch+1}/{epochs} - Loss: {np.mean(batch_losses):.4f}")

    # 2. Evaluate on specific zones for 1:1 comparison with Federated/Local
    model.eval()
    rmses = []
    with torch.no_grad():
        for zone in ZONES:
            preds = model(torch.Tensor(X_test_dict[zone])).detach().numpy()
            true = y_test_dict[zone]
            zone_rmse = np.sqrt(mean_squared_error(true.flatten(), preds.flatten()))
            rmses.append(zone_rmse)
            print(f"[{zone}] Centralized {model_name} RMSE: {zone_rmse:.4f}")

    return np.mean(rmses)
    
def run_local(model_class, model_name, epochs):
    print(f"\n--- Running Local Training: {model_name} ---")
    rmses = []
    all_losses = {}
    for zone in ZONES:
        # Unpacking 5 values as required by current data_utils logic
        X, y, _, _, _, temps = prepare_zonal_data(zone, LOAD_FILE, f"{BASE_PATH}/{zone}_TX_Total.xlsx")
        split = int(0.8 * len(X))
        model = model_class(input_dim=12) # Explicitly 9
        train_loader = get_dataloader(X[:split], y[:split], shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        zone_epoch_losses = []
          
        model.train()
        for epoch in range(epochs):
            batch_losses = []
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by) # nsure targets are the correct shape
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            avg_epoch_loss = np.mean(batch_losses)
            zone_epoch_losses.append(avg_epoch_loss)
            print(f"[{zone}] Epoch {epoch+1}/{epochs} - Loss: {avg_epoch_loss:.4f}")
        all_losses[zone] = zone_epoch_losses
        
        model.eval()
        with torch.no_grad():
            preds = model(torch.Tensor(X[split:])).detach().numpy()
            true = y[split:]
            rmses.append(np.sqrt(mean_squared_error(true.flatten(), preds.flatten())))
        print(f"[{zone}] Local {model_name} RMSE: {rmses[-1]:.4f}")

        # [RAI UPDATE] Run Fairness and SHAP audits locally
        run_fairness_audit(true, preds, temps[split:], f"Local {model_name}", zone)
        if zone == "NCENT": # Run SHAP on one zone to demonstrate explainability without long compute
            run_shap_analysis(model, torch.Tensor(X[:split]), torch.Tensor(X[split:]), f"Local {model_name}")
        
    plt.figure(figsize=(8,5))
    for zone, losses in all_losses.items():
        plt.plot(losses, label=f"{zone} Training Loss")
    plt.title(f"Local Training Loss Curves: {model_name}")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    clean_name = model_name.lower().replace(' ', '_')
    plt.savefig(f"{BASE_PATH}/local_loss_curve_{clean_name}.png")
    plt.close()
    print(f"Local loss plot saved for {model_name}!")

    return np.mean(rmses)

def run_federated(model_class, model_name, rounds):
    print(f"\n--- Running Federated Training: {model_name} ---")
    def client_fn(context):
        cid = int(context.node_config["partition-id"])
        zone = ZONES[cid]
        X, y, _, _, _, _= prepare_zonal_data(zone, LOAD_FILE, f"{BASE_PATH}/{zone}_TX_Total.xlsx")
        s = int(0.8 * len(X))
        return ERCOTClient(model_class(input_dim=12), get_dataloader(X[:s], y[:s]), get_dataloader(X[s:], y[s:])).to_client()

    clean_name = model_name.lower().replace(' ', '_')
    strategy = SaveModelStrategy(
        model_class=model_class,
        save_name=f"fed_{clean_name}_weights",
        min_fit_clients=3, 
        min_available_clients=3, 
        evaluate_metrics_aggregation_fn=aggregate_metrics
    )
    history = fl.simulation.start_simulation(client_fn=client_fn, num_clients=3, config=fl.server.ServerConfig(num_rounds=rounds), strategy=strategy, ray_init_args={
        #"num_cpus": 1,
        "include_dashboard": False,
        "configure_logging": True,
        "logging_level": logging.ERROR, # silences the RpcError noise
        "log_to_driver": False, 
    })

    #Plot federated loss
    rounds, loss_values = zip(*history.losses_distributed)
    plt.figure(figsize=(8,5))
    plt.plot(rounds, loss_values, marker='o', color='blue', label='Global Fed Loss')
    plt.title(f"Federated Learning Convergence: {model_name}")
    plt.xlabel("Communication Round")
    plt.ylabel("Aggregated MSE Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha = 0.6)
    plt.savefig(f"{BASE_PATH}/federated_loss_curve_{clean_name}.png")
    plt.close()
    print(f"Federated loss plot saved for {model_name}!")

    # Save the entire history object to disk
    with open(f"{BASE_PATH}/federated_history_{clean_name}.pkl", "wb") as f:
        pickle.dump(history, f)
    print(f"Training history saved to federated_history_{clean_name}.pkl")
    final_fed_rmse = history.metrics_distributed['rmse'][-1][1]
    final_fed_mape = history.metrics_distributed['mape'][-1][1]

    print(f"Federated {model_name} Global RMSE: {final_fed_rmse:.4f}")
    print(f"Federated {model_name} Global MAPE: {final_fed_mape:.2f}%")

    # [RAI UPDATE] Post-Training Global Audit & Governance
    global_model = model_class(input_dim=12)
    global_model.load_state_dict(torch.load(f"{BASE_PATH}/fed_{clean_name}_weights.pt", weights_only=True))
    global_model.eval()

    fairness_gaps = {}
    print(f"\n--- Initiating Global RAI Audit for {model_name} ---")
    # Grab data from one zone (e.g., NCENT) to run global SHAP explainer
    X_sample, y_sample, _, _, _, _ = prepare_zonal_data("NCENT", LOAD_FILE, f"{BASE_PATH}/NCENT_TX_Total.xlsx")
    split_sample = int(0.8 * len(X_sample))
    run_shap_analysis(global_model, torch.Tensor(X_sample[:split_sample]), torch.Tensor(X_sample[split_sample:]), f"Federated {model_name}")

    with torch.no_grad():
        for zone in ZONES:
            X, y, _, _, _, temps = prepare_zonal_data(zone, LOAD_FILE, f"{BASE_PATH}/{zone}_TX_Total.xlsx")
            split = int(0.8 * len(X))
            preds = global_model(torch.Tensor(X[split:])).detach().numpy()
            gap = run_fairness_audit(y[split:], preds, temps[split:], f"Federated {model_name}", zone)
            fairness_gaps[zone] = float(gap)

    generate_model_card(f"Federated {model_name}", final_fed_rmse, final_fed_mape, fairness_gaps)

    return final_fed_rmse

def run_baselines():
    print("\n--- Running Statistical Baselines ---")
    p_rmses, lr_rmses, a_rmses = [], [], []
    for zone in ZONES:
        X, y, _, _, _, _= prepare_zonal_data(zone, LOAD_FILE, f"{BASE_PATH}/{zone}_TX_Total.xlsx")
        split = int(0.8 * len(X))
        
        # Persistence
        p_pred = X[split:, :, 0]
        p_rmses.append(np.sqrt(mean_squared_error(y[split:].flatten(), p_pred.flatten())))
        
        # Scikit Linear Regression
        lr_pred = train_linear_baseline(X[:split], y[:split], X[split:])
        lr_rmses.append(np.sqrt(mean_squared_error(y[split:].flatten(), lr_pred.flatten())))
        
        # ARIMA (Predicting only next 24h for speed)
        a_pred = get_arima_baseline(y[:split, 0], horizon=24)
        a_rmses.append(np.sqrt(mean_squared_error(y[split], a_pred)))

    return np.mean(p_rmses), np.mean(lr_rmses), np.mean(a_rmses)

if __name__ == "__main__":

    architectures = [
        ("BiLSTM", BiLSTMForecaster),
        ("Standard LSTM", StandardLSTMForecaster),
        ("PyTorch Linear", LinearForecaster)
    ]

    results = {}
    
    # 1. Run Statistical Baselines
    p_rmse, lr_rmse, a_rmse = run_baselines()
    results['Persistence'] = p_rmse
    results['Sklearn Linear'] = lr_rmse
    results['ARIMA'] = a_rmse

    # 2. Run Comparative Audit (Centralized vs Local vs Federated)
    for name, model_cls in architectures:
        results[f'Centralized {name}'] = run_centralized(model_cls, name, epochs=50) 
        results[f'Local {name}'] = run_local(model_cls, name, epochs=50) 
        results[f'Federated {name}'] = run_federated(model_cls, name, rounds=50)

    # 3. Print Final Thesis Leaderboard
    print("\n" + "="*60)
    print("FINAL ARCHITECTURE LEADERBOARD (AVG RMSE) ")
    print("="*60)
    # Sort results by lowest error
    for name, error in sorted(results.items(), key=lambda item: item[1]):
        print(f"{name:<25}: {error:.4f}")
    print("="*60)
