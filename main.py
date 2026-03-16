import flwr as fl
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from flwr.common import parameters_to_ndarrays
from sklearn.metrics import mean_squared_error
from data_utils import prepare_zonal_data, get_dataloader, ZONES, BASE_PATH
from model import BiLSTMForecaster
from client import ERCOTClient
import logging
import os
import pickle
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning, module="flwr") # ignores the specific DeprecationWarning from Flower

# silence the "RpcError" and metrics exporter noise
os.environ["RAY_metrics_export_binaries_path"] = ""
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO"] = "0"

LOAD_FILE = f"{BASE_PATH}/Demand_Data.xlsx"
#s_target, s_feat = get_global_scalers(LOAD_FILE, BASE_PATH)

def aggregate_metrics(metrics):
    if not metrics: return {}
    total = sum([n for n, _ in metrics])
    return {"rmse": sum([n * m["rmse"] for n, m in metrics]) / total,
            "base_rmse": sum([n * m["base_rmse"] for n, m in metrics]) / total}

# Custom Strategy to Save the Global Model
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Call the parent class to aggregate the weights normally
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        
        # Save the model if aggregation is successful
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            model = BiLSTMForecaster(input_dim=10)
            # Map the aggregated numpy arrays back to the PyTorch model's state dictionary
            params_dict = zip(model.state_dict().keys(), ndarrays)
            state_dict = {k: torch.tensor(v) for k, v in params_dict}
            model.load_state_dict(state_dict, strict=True)
            
            # Save to disk
            torch.save(model.state_dict(), f"{BASE_PATH}/final_model_weights.pt")
            print(f"💾 Global model saved to disk at round {server_round}!")
            
        return aggregated_parameters, aggregated_metrics
    
def run_local():
    print("--- Running Local Training Baseline ---")
    rmses = []
    all_losses = {}
    for zone in ZONES:
        # Unpacking 5 values as required by current data_utils logic
        X, y, _, _, _ = prepare_zonal_data(zone, LOAD_FILE, f"{BASE_PATH}/{zone}_TX_Total.xlsx")
        split = int(0.8 * len(X))
        model = BiLSTMForecaster(input_dim=10) # Explicitly 9
        train_loader = get_dataloader(X[:split], y[:split], shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.MSELoss()

        zone_epoch_losses = []
          
        model.train()
        for epoch in range(10):
            batch_losses = []
            for bx, by in train_loader:
                optimizer.zero_grad()

                loss = criterion(model(bx), by) # Ensure targets are the correct shape
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())
            avg_epoch_loss = np.mean(batch_losses)
            zone_epoch_losses.append(avg_epoch_loss)
            print(f"[{zone}] Epoch {epoch+1}/10 - Loss: {avg_epoch_loss:.4f}")
        all_losses[zone] = zone_epoch_losses
        
        model.eval()
        with torch.no_grad():
            preds = model(torch.Tensor(X[split:])).detach().numpy()
            true = y[split:]
            # y is [samples, 24, 1] so squeeze it for evaluation
            rmses.append(np.sqrt(mean_squared_error(true.flatten(), preds.flatten())))
        
        plt.figure(figsize=(8,5))
        for zone, losses in all_losses.items():
            plt.plot(losses, label=f"{zone} Training Loss")
        plt.title("Local Training Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(f"{BASE_PATH}/local_loss_curve.png")
        print("Local loss plot saved!")
    return np.mean(rmses)

def run_federated():
    print("--- Running Federated Simulation ---")
    def client_fn(context):
        cid = int(context.node_config["partition-id"])
        zone = ZONES[cid]
        X, y, _, _, _= prepare_zonal_data(zone, LOAD_FILE, f"{BASE_PATH}/{zone}_TX_Total.xlsx")
        s = int(0.8 * len(X))
        return ERCOTClient(BiLSTMForecaster(input_dim=10), get_dataloader(X[:s], y[:s]), get_dataloader(X[s:], y[s:])).to_client()

    strategy = SaveModelStrategy(
        min_fit_clients=3, 
        min_available_clients=3, 
        evaluate_metrics_aggregation_fn=aggregate_metrics
    )
    history = fl.simulation.start_simulation(client_fn=client_fn, num_clients=3, config=fl.server.ServerConfig(num_rounds=50), strategy=strategy, ray_init_args={
        #"num_cpus": 1,
        "include_dashboard": False,
        "configure_logging": True,
        "logging_level": logging.ERROR, # This silences the RpcError noise
        "log_to_driver": False, 
    })

    #Plot federated loss
    rounds, loss_values = zip(*history.losses_distributed)
    plt.figure(figsize=(8,5))
    plt.plot(rounds, loss_values, marker='o', color='red', label='Global Fed Loss')
    plt.title("Federated Learning Convergence (Global Model)")
    plt.xlabel("Communication Round")
    plt.ylabel("aggregated MSE Loss")
    plt.legend()
    plt.grid(True, linestyle='--', alpha = 0.6)
    plt.savefig(f"{BASE_PATH}/federated_loss_curve.png")
    print("Federated loss plot saved!")

    # Save the entire history object to disk
    with open(f"{BASE_PATH}/federated_history.pkl", "wb") as f:
        pickle.dump(history, f)
    print("✅ Training history saved to federated_history.pkl")
    final_base_rmse = history.metrics_distributed['base_rmse'][-1][1]
    final_fed_rmse = history.metrics_distributed['rmse'][-1][1]

    print("\n" + "="*30)
    print("FINAL PERFORMANCE COMPARISON")
    print("="*30)
    print(f"1. ERCOT Baseline RMSE:    {final_base_rmse:.4f}")
    print(f"2. Local AI RMSE:          {l_rmse:.4f}")
    print(f"3. Federated AI RMSE:      {final_fed_rmse:.4f}")
    print("-" * 30)

    improvement = ((final_base_rmse - final_fed_rmse) / final_base_rmse) * 100
    print(f"FL Improvement over ERCOT: {improvement:.2f}%")

    return history.metrics_distributed['rmse'][-1][1]

if __name__ == "__main__":
    l_rmse = run_local()
    f_rmse = run_federated()

    print(f"\n✅ RESULTS COMPLETED:\nLocal Avg RMSE: {l_rmse:.4f}\nFederated RMSE: {f_rmse:.4f}")
    # Extract the baseline from the last round of FL history
    # history.metrics_distributed['base_rmse'] is a list of (round, value)
    
