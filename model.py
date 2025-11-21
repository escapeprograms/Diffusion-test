import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from data import create_input

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class GaussianFourierProjection(nn.Module):
    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Random weights fixed at initialization
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
    
    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class ResnetBlock(nn.Module):
    def __init__(self, dim, time_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim + time_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, time_emb):
        # Concatenate time to input
        h = torch.cat([x, time_emb], dim=1)
        # Residual connection: output + input
        return x + self.mlp(h)

class FiLMResBlock(nn.Module):
    def __init__(self, dim, time_dim, dropout=0.1):
        super().__init__()
        
        # Standard spatial processing
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        
        # Project time embedding to (Scale, Shift) for each feature
        self.time_proj = nn.Linear(time_dim, dim * 2)
        
        # Initialize time projection to 0 so it starts as Identity function
        # (scale=0, shift=0) -> y = x * (1+0) + 0 = x
        # nn.init.zeros_(self.time_proj.weight)
        # nn.init.zeros_(self.time_proj.bias)

    def forward(self, x, time_emb):
        # 1. Process x
        h = self.mlp(x)
        
        # 2. Get Scale and Shift from Time
        # style shape: (Batch, 2 * dim)
        style = self.time_proj(time_emb)
        scale, shift = style.chunk(2, dim=1)
        
        # 3. Modulate (FiLM)
        # This forces the time signal to modify the features
        h = h * (1 + scale) + shift
        
        # 4. Residual
        return x + h

class NoisePredictionMLP(nn.Module):
    """
    The Noise Prediction Network (epsilon-predictor) for a 2D diffusion model.
    It takes the noisy data point (x, y) and the time step (t) as input 
    and predicts the noise vector (epsilon_x, epsilon_y) added at that time.

    Input size: 3 (x, y, t)
    Output size: 2 (predicted noise epsilon_x, epsilon_y)
    """
    def __init__(self, hidden_size=512, num_layers=6):
        """
        Initializes the Multi-Layer Perceptron (MLP).
        
        Args:
            hidden_size (int): The number of neurons in the hidden layers.
        """
        super().__init__()

        #time embedding
        time_dim = 256
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        input_dim = 2 + time_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.layers = nn.ModuleList([
            FiLMResBlock(hidden_size, time_dim) for _ in range(num_layers)
        ])
        self.final = nn.Linear(hidden_size, 2)


    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass.

        Args:
            x_in (torch.Tensor): Input tensor of shape (batch_size, 3), 
                                 where 3 corresponds to (noisy_x, noisy_y, t).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 2), 
                          corresponding to the predicted noise vector (epsilon_x, epsilon_y).
        """
        coords = x_in[:, :2]
        t = x_in[:, 2]
        
        time_embed = self.time_mlp(t)
        # Linear layer 1 followed by Swish activation (often better than ReLU in diffusion models)
        x = F.silu(self.fc1(torch.cat([coords, time_embed], dim=1)))
        for layer in self.layers:
            x = layer(x, time_embed)
        
        # Output layer: no activation (predicting noise, which is unbounded)
        output = self.final(x)
        
        # The output 'output' is the predicted noise: [epsilon_x, epsilon_y]
        return output


@torch.no_grad()
def sample_ddim(model, n_samples, alpha_bars, inference_steps=20, eta=0.0, history=False):
    """
    DDIM Sampling.
    
    Args:
        inference_steps: Number of steps to take (can be less than training steps T).
        eta: 0.0 = Deterministic DDIM (recommended). 
             1.0 = Same behavior as standard DDPM.
    """
    
    model.eval()
    total_train_steps = len(alpha_bars)
    
    # 1. Select the specific time steps we will use
    # e.g., if T=100 and inference_steps=10, we might use [90, 80, ..., 0]
    times = torch.linspace(0, total_train_steps - 1, inference_steps).long()
    times = list(reversed(times.tolist())) # Reverse to go from noisy -> clean
    
    # Create pairs of (current_step, next_step)
    # e.g., [(90, 80), (80, 70), ..., (10, 0), (0, -1)]
    time_pairs = list(zip(times[:-1], times[1:])) + [(times[-1], -1)]
    
    # Start from pure noise
    x = torch.randn(n_samples, 2)

    #track history
    history = [x.numpy().copy()]
    
    print(f"DDIM Sampling with {inference_steps} steps...")
    
    for t_curr, t_next in time_pairs:
        # Broadcast time for the batch
        t_tensor = torch.full((n_samples, 1), t_curr, dtype=torch.float32)
        
        # Predict noise
        model_input = torch.cat([x, t_tensor], dim=1)
        predicted_noise = model(model_input)
        
        # Get alpha_bar values for current and next step
        alpha_bar_t = alpha_bars[t_curr]
        alpha_bar_t_next = alpha_bars[t_next] if t_next >= 0 else 1.0
        
        # --- DDIM Update Formula ---
        
        # 1. Estimate clean x0 (predicted original point)
        sqrt_alpha_bar_t = np.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = np.sqrt(1 - alpha_bar_t)
        pred_x0 = (x - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
        
        # 2. Calculate "direction" to point x_{t-1}
        # (This sigma is 0 when eta=0, making it deterministic)
        sigma_t = eta * np.sqrt((1 - alpha_bar_t_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_next))
        dir_xt = np.sqrt(1 - alpha_bar_t_next - sigma_t**2) * predicted_noise
        
        # 3. Update x
        if sigma_t > 0:
            noise = torch.randn_like(x)
        else:
            noise = 0
            
        x = np.sqrt(alpha_bar_t_next) * pred_x0 + dir_xt + sigma_t * noise
        if history:
            history.append(x.numpy().copy())
            
    return x.numpy(), history


@torch.no_grad()
def vector_field_ddim(model, alpha_bars, inference_steps=20, x_values=[], y_values=[]):
    model.eval()
    total_train_steps = len(alpha_bars)
    
    # 1. Select the specific time steps we will use
    # e.g., if T=100 and inference_steps=10, we might use [90, 80, ..., 0]
    times = torch.linspace(0, total_train_steps - 1, inference_steps).long()
    times = list(reversed(times.tolist())) # Reverse to go from noisy -> clean
    
    # Create pairs of (current_step, next_step)
    # e.g., [(90, 80), (80, 70), ..., (10, 0), (0, -1)]
    time_pairs = list(zip(times[:-1], times[1:])) + [(times[-1], -1)]
    
    # points to get vectors for
    points = torch.stack([torch.tensor(x_values), torch.tensor(y_values)],dim=1)

    history = []
    
    print(f"DDIM Sampling with {inference_steps} steps...")
    
    for t_curr, t_next in time_pairs:
        # Broadcast time for the batch
        t_tensor = torch.full((len(x_values), 1), t_curr, dtype=torch.float32)
        
        # Predict noise
        model_input = torch.cat([points, t_tensor], dim=1)
        predicted_noise = model(model_input)
        
        # Get alpha_bar values for current and next step
        alpha_bar_t = alpha_bars[t_curr]
        alpha_bar_t_next = alpha_bars[t_next] if t_next >= 0 else 1.0
        
        # --- DDIM Update Formula ---
        
        # 1. Estimate clean x0 (predicted original point)
        sqrt_alpha_bar_t = np.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = np.sqrt(1 - alpha_bar_t)
        pred_x0 = (points - sqrt_one_minus_alpha_bar_t * predicted_noise) / sqrt_alpha_bar_t
        
        # 2. Calculate "direction" to point x_{t-1}
        # (This sigma is 0 when eta=0, making it deterministic)
        eta=0
        sigma_t = eta * np.sqrt((1 - alpha_bar_t_next) / (1 - alpha_bar_t) * (1 - alpha_bar_t / alpha_bar_t_next))
        dir_xt = np.sqrt(1 - alpha_bar_t_next - sigma_t**2) * predicted_noise
        
        history.append(dir_xt.numpy())
            
        # points = np.sqrt(alpha_bar_t_next) * pred_x0 + dir_xt + sigma_t * noise
            
    return history