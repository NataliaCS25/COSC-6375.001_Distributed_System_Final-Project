import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ERCOTClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.MSELoss()
        #self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)
        # Drop the learning rate by half every 5 rounds
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=5, gamma=0.5)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)


    def fit(self, parameters, config):
        self.set_parameters(parameters)
        # Save a copy of the global parameters for the proximal term
        global_params = [p.detach().clone() for p in self.model.parameters()]
        
        self.model.train()
        mu = 0.01  # The proximal term constant (start small)
        
        for _ in range(10):  # Reduced from 5 to 2 epochs to prevent drift
            for batch_x, batch_y in self.train_loader:
                self.optimizer.zero_grad()
                proximal_term = 0.0
                
                # Calculate the L2 difference between local and global weights
                for name, param in self.model.named_parameters():
                    # Note: You'll need a mapping if names don't match, 
                    # but for this simple setup, we can use the list:
                    pass 
                
                # Simplified Proximal Loss implementation:
                output = self.model(batch_x)
                loss = self.criterion(output, batch_y)
                
                # Add the penalty: mu/2 * ||w - w_t||^2
                for local_p, global_p in zip(self.model.parameters(), global_params):
                    proximal_term += (local_p - global_p).pow(2).sum()
                
                total_loss = loss + (mu / 2) * proximal_term
                total_loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        all_p, all_t = [], []
        with torch.no_grad():
            for bx, by in self.val_loader:
                all_p.append(self.model(bx).numpy())
                all_t.append(by.numpy())
        all_p = np.concatenate(all_p)
        all_t = np.concatenate(all_t)

        ai_rmse = np.sqrt(mean_squared_error(all_t, all_p))
        baseline_rmse = np.sqrt(mean_squared_error(all_t, np.zeros_like(all_t)))
        #mse = mean_squared_error(all_t, all_p)
        return float(ai_rmse), len(self.val_loader.dataset), {"mae": float(mean_absolute_error(all_t, all_p)), "rmse": float(ai_rmse), "base_rmse": float(baseline_rmse)}
