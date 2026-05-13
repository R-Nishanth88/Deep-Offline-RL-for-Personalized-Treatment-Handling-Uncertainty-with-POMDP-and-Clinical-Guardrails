import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.logger import setup_logger

logger = setup_logger("latent_model")

class LatentVAE(nn.Module):
    """
    Variational Autoencoder (VAE) to estimate hidden patient condition z_t.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(LatentVAE, self).__init__()
        
        # Encoder
        self.encoder_fc = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_fc = nn.Linear(latent_dim, hidden_dim)
        self.decoder_out = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        h = F.relu(self.encoder_fc(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.decoder_fc(z))
        return torch.sigmoid(self.decoder_out(h)) # Assuming normalized inputs

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss_function(recon_x, x, mu, logvar):
    """
    VAE Loss = Reconstruction Loss + KL Divergence.
    """
    BCE = F.mse_loss(recon_x, x, reduction='sum')
    # KL divergence: 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

class VAETrainer:
    def __init__(self, model, lr=1e-3, device='cpu'):
        self.model = model.to(device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.device = device
        
    def train_step(self, x):
        self.model.train()
        self.optimizer.zero_grad()
        
        recon_batch, mu, logvar = self.model(x.to(self.device))
        loss = vae_loss_function(recon_batch, x.to(self.device), mu, logvar)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()
