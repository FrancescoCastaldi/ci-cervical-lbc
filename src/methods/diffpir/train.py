import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.dataset import load_config, LBCDataset
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for clean in train_loader:
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
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for clean in val_loader:
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
        val_loss /= len(val_dataset)

        scheduler.step()
        print(f"Epoch {epoch:2d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    if save_path is None:
        save_path = Path(__file__).resolve().parent / "weights" / "ddpm_lbc.pt"
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
    train()
