import torch
import torch.nn as nn
import torch.optim as optim
from src.utils.logger import setup_logger

logger = setup_logger("temporal_model")

class TemporalEncoder(nn.Module):
    """
    LSTM-based encoder to capture temporal dependencies in patient trajectories.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=2):
        super(TemporalEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        # Take the last hidden state
        out = out[:, -1, :]
        embedding = self.fc(out)
        return embedding

class TemporalTrainer:
    def __init__(self, model, lr=1e-3, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.MSELoss() # Example: reconstruct the last state or predict next
        self.device = device
        
    def train_step(self, x, y):
        self.model.train()
        self.optimizer.zero_grad()
        
        embedding = self.model(x.to(self.device))
        loss = self.criterion(embedding, y.to(self.device))
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save_checkpoint(self, path):
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model checkpoint saved to {path}")
