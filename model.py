import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, encoder_outputs):
        # encoder_outputs: [batch, seq_len, hidden_dim]
        energy = self.projection(encoder_outputs) # [batch, seq_len, 1]
        weights = F.softmax(energy.squeeze(-1), dim=1) # [batch, seq_len]
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1) # [batch, hidden_dim]
        return outputs, weights

class BiLSTMForecaster(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=128, num_layers=2, forecast_horizon=24):
        super().__init__()
        # input_dim=9: [Res, Temp, Hr_Sin, Hr_Cos, Day_Sin, Day_Cos, 3 Region IDs]
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                            batch_first=True, bidirectional=True, dropout=0.2)
        self.attention = SelfAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, forecast_horizon) 

    def forward(self, x):
        lstm_out, _ = self.lstm(x) # [batch, 168, hidden*2]
        attn_out, weights = self.attention(lstm_out) # [batch, hidden*2]
        x = self.dropout(attn_out)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
