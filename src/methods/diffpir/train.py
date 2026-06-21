import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from src.utils import get_device
from src.data.dataset import load_config, LBCDataset, PROJECT_ROOT
from src.methods.diffpir.model import LightUNet


def get_ddpm_schedule(num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
    betas = torch.linspace(beta_start, beta_end, num_timesteps)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_cumprod


def train(epochs=30, lr=1e-4, batch_size=4, num_timesteps=1000, subset=100, save_path=None):
    config = load_config()

    train_dataset = LBCDataset("data/splits/train.txt", image_size=config["dataset"]["image_size"])
    val_dataset = LBCDataset("data/splits/val.txt", image_size=config["dataset"]["image_size"])

    if subset and subset < len(train_dataset):
        train_dataset.paths = train_dataset.paths[:subset]
        print(f"Uso subset di {subset} immagini per training")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = get_device()
    print(f"Dispositivo: {device}")
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} immagini")
    print(f"Timesteps: {num_timesteps}")

    model = LightUNet(in_channels=3, out_channels=3, base_channels=32).to(device)
    print(f"Parametri: {sum(p.numel() for p in model.parameters()):,}")
    
    betas, alphas, alphas_cumprod = get_ddpm_schedule(num_timesteps)
    betas = betas.to(device)
    alphas = alphas.to(device)
    alphas_cumprod = alphas_cumprod.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    epoch_bar = tqdm(range(1, epochs + 1), desc="Training", unit="epoch")
    for epoch in epoch_bar:
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch", leave=False)
        for clean in train_bar:
            clean = clean.to(device)
            batch_size = clean.size(0)
            
            t = torch.randint(0, num_timesteps, (batch_size,), device=device)
            noise = torch.randn_like(clean)
            
            sqrt_alpha_bar = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
            sqrt_one_minus_alpha_bar = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
            
            noisy = sqrt_alpha_bar * clean + sqrt_one_minus_alpha_bar * noise
            
            optimizer.zero_grad()
            pred_noise = model(noisy, t)
            loss = criterion(pred_noise, noise)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch_size
            train_bar.set_postfix(loss=loss.item())
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        val_bar = tqdm(val_loader, desc=f"Val {epoch}/{epochs}", unit="batch", leave=False)
        with torch.no_grad():
            for clean in val_bar:
                clean = clean.to(device)
                batch_size = clean.size(0)
                
                t = torch.randint(0, num_timesteps, (batch_size,), device=device)
                noise = torch.randn_like(clean)
                
                sqrt_alpha_bar = alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_bar = (1 - alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
                
                noisy = sqrt_alpha_bar * clean + sqrt_one_minus_alpha_bar * noise
                pred_noise = model(noisy, t)
                loss = criterion(pred_noise, noise)
                val_loss += loss.item() * batch_size
                val_bar.set_postfix(loss=loss.item())
        val_loss /= len(val_dataset)

        scheduler.step()
        epoch_bar.set_postfix(train=train_loss, val=val_loss)

    if save_path is None:
        save_path = PROJECT_ROOT / "src" / "methods" / "diffpir" / "weights" / "ddpm_lbc.pt"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_timesteps": num_timesteps,
        "betas": betas.cpu(),
        "alphas": alphas.cpu(),
        "alphas_cumprod": alphas_cumprod.cpu(),
    }, str(save_path))
    print(f"Modello salvato in: {save_path}")


if __name__ == "__main__":
    train(epochs=50, num_timesteps=1000, subset=0, lr=1e-4, batch_size=4)
