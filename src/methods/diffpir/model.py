import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalTimestepEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)
        
        self.time_mlp = None
        if time_emb_dim:
            self.time_mlp = nn.Sequential(
                nn.Linear(time_emb_dim, out_ch),
                nn.SiLU()
            )
        
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb=None):
        h = self.norm1(self.conv1(x))
        h = F.silu(h)
        
        if self.time_mlp is not None and t_emb is not None:
            h = h + self.time_mlp(t_emb).unsqueeze(-1).unsqueeze(-1)
        
        h = self.norm2(self.conv2(h))
        h = F.silu(h)
        
        return h + self.skip(x)


class LightUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, base_channels=32):
        super().__init__()
        
        time_emb_dim = base_channels * 4
        
        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        self.init_conv = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        self.down1 = ResBlock(base_channels, base_channels, time_emb_dim)
        self.down2 = ResBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down3 = ResBlock(base_channels * 2, base_channels * 4, time_emb_dim)
        
        self.pool1 = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)
        self.pool2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)
        
        self.mid1 = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        self.mid2 = ResBlock(base_channels * 4, base_channels * 4, time_emb_dim)
        
        self.up1 = ResBlock(base_channels * 2 + base_channels * 2, base_channels * 2, time_emb_dim)
        self.up2 = ResBlock(base_channels + base_channels, base_channels, time_emb_dim)
        self.up3 = ResBlock(base_channels, base_channels, time_emb_dim)
        
        self.unpool1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.unpool2 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        
        self.final_conv = nn.Conv2d(base_channels, out_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_embed(t)
        
        x = self.init_conv(x)
        
        h1 = self.down1(x, t_emb)
        h1_pool = self.pool1(h1)
        
        h2 = self.down2(h1_pool, t_emb)
        h2_pool = self.pool2(h2)
        
        h3 = self.down3(h2_pool, t_emb)
        
        h = self.mid1(h3, t_emb)
        h = self.mid2(h, t_emb)
        
        h = self.unpool1(h)
        h = torch.cat([h, h2], dim=1)
        h = self.up1(h, t_emb)
        
        h = self.unpool2(h)
        h = torch.cat([h, h1], dim=1)
        h = self.up2(h, t_emb)
        
        h = self.up3(h, t_emb)
        
        return self.final_conv(h)
