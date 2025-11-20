#generate data from sample
import numpy as np
import torch

#make the "steps" efficient
def setup_noise_schedule(time_steps: int = 1000):
    # ...
    # 1. Define the variance schedule (beta_t)
    beta_start = 1e-4
    beta_end = 0.02
    betas = np.linspace(beta_start, beta_end, time_steps)
    
    # 2. Calculate alpha_t and alpha_bar_t
    alphas = 1.0 - betas
    alpha_bars = np.cumprod(alphas) # This is the crucial cumulative product
    
    return alpha_bars, betas

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0.0001, 0.9999).numpy()
    alpha_bars = np.cumprod(1.0 - betas)
    return alpha_bars, betas


def create_input(x_points, y_points, t):
    t_tensor = torch.tensor([t]*len(x_points)).float().unsqueeze(1) # Convert to (batch_size, 1) tensor
    points = np.stack([x_points, y_points], axis=1)
    model_input_x_t = torch.from_numpy(points).float()
    input = torch.cat([model_input_x_t, t_tensor], dim=1) # (Shape: batch_size, 3)
    
    return input

def create_batch(alpha_bars, x_points, y_points, batch_size):
    orig_points = np.stack([x_points, y_points], axis=1)
    #sample points from distr
    indices = np.random.choice(len(orig_points), size=batch_size, replace=True)
    x_0 = orig_points[indices] #original points
    eps = np.random.normal(0, 1, len(x_0)*2) #noise (for 2 dims)
    eps = eps.reshape(-1,2)
    
    t = np.random.randint(0, len(alpha_bars), len(x_0)) #time steps
    # print(x_0.shape, eps.shape, t.shape)

    #x_t = sqrt(alpha_bar[t])x_0 + sqrt(1-alhpa_bar[t])N(0,1)
    a1 = np.sqrt(alpha_bars[t])
    a2 = np.sqrt(1.0-alpha_bars[t]) #(batch,)

    a1 = a1.reshape(-1,1)
    a2 = a2.reshape(-1,1) #(batch, 1)
    x_t = a1*x_0 + a2 * eps

    #convert to training data
    t_tensor = torch.from_numpy(t).float().unsqueeze(1) # Convert to (batch_size, 1) tensor
    
    model_input_x_t = torch.from_numpy(x_t).float()
    # input = torch.cat([model_input_x_t, t_tensor], dim=1) # (Shape: batch_size, 3)
    output = torch.from_numpy(eps).float() # (Shape: batch_size, 2)
    
    return model_input_x_t, t_tensor, output

