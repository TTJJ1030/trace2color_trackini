"""
CBAM-enhanced Multi-Scale UNet for radar point quality classification.

Innovation: Multi-layer Temporal-Spatial Tensor Encoding (MTSTE) with
CBAM attention-guided segmentation network.

Input: 8-channel tensor [B, 8, H, W]
Output: per-pixel confidence map [B, 2, H, W] (target vs clutter)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ChannelAttention(nn.Module):
    """Channel attention block (CBAM channel part)."""

    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(channels // reduction, 1), channels),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.mlp(self.gap(x))
        mx = self.mlp(self.gmp(x))
        scale = self.sigmoid(avg + mx).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class SpatialAttention(nn.Module):
    """Spatial attention block (CBAM spatial part)."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        combined = torch.cat([avg, mx], dim=1)
        scale = self.sigmoid(self.conv(combined))
        return x * scale


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, channels: int, reduction: int = 8, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(spatial_kernel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ca(x)
        x = self.sa(x)
        return x


class ConvBlock(nn.Module):
    """Double conv block: Conv-BN-ReLU × 2."""

    def __init__(self, in_ch: int, out_ch: int, use_cbam: bool = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(out_ch) if use_cbam else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cbam(self.conv(x))


class DownBlock(nn.Module):
    """Encoder block: max pool + conv."""

    def __init__(self, in_ch: int, out_ch: int, use_cbam: bool = True):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_ch, out_ch, use_cbam=use_cbam)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class UpBlock(nn.Module):
    """Decoder block: bilinear upsample + conv with skip connection."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle size mismatch
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class FPNFusion(nn.Module):
    """Feature Pyramid Network-style multi-scale feature fusion."""

    def __init__(self, channels: list):
        super().__init__()
        self.laterals = nn.ModuleList([
            nn.Conv2d(c, channels[0], 1) for c in channels
        ])
        self.outputs = nn.ModuleList([
            nn.Conv2d(channels[0], channels[0], 3, padding=1) for _ in channels
        ])

    def forward(self, features: list) -> list:
        # Top-down pathway
        lat = [l(f) for l, f in zip(self.laterals, features)]
        for i in range(len(lat) - 2, -1, -1):
            upsampled = F.interpolate(lat[i + 1], size=lat[i].shape[2:],
                                      mode='bilinear', align_corners=True)
            lat[i] = lat[i] + upsampled
        return [o(l) for o, l in zip(self.outputs, lat)]


class CBAMUNet(nn.Module):
    """
    Multi-Scale CBAM-UNet for radar point quality classification.

    Input: [B, 8, H, W] multi-layer tensor
    Output: [B, 2, H, W] per-pixel logits (target vs clutter)
    """

    def __init__(self, in_channels: int = 8, n_classes: int = 2,
                 base_ch: int = 32):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, base_ch, use_cbam=False)
        self.enc2 = DownBlock(base_ch, base_ch * 2, use_cbam=True)
        self.enc3 = DownBlock(base_ch * 2, base_ch * 4, use_cbam=True)
        self.enc4 = DownBlock(base_ch * 4, base_ch * 8, use_cbam=True)

        # Bottleneck
        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 8, use_cbam=True)

        # FPN multi-scale fusion (all outputs have base_ch*8 channels)
        fpn_out_ch = base_ch * 8
        self.fpn = FPNFusion([base_ch * 8, base_ch * 4, base_ch * 2])

        # Decoder: in_ch = output of previous decoder, skip_ch = fpn output (all fpn_out_ch)
        self.dec3 = UpBlock(fpn_out_ch, fpn_out_ch, base_ch * 4)
        self.dec2 = UpBlock(base_ch * 4, fpn_out_ch, base_ch * 2)
        self.dec1 = UpBlock(base_ch * 2, base_ch, base_ch)

        # Output
        self.out_conv = nn.Conv2d(base_ch, n_classes, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e1 = self.enc1(x)       # [B, 32, H, W]
        e2 = self.enc2(e1)      # [B, 64, H/2, W/2]
        e3 = self.enc3(e2)      # [B, 128, H/4, W/4]
        e4 = self.enc4(e3)      # [B, 256, H/8, W/8]

        # Bottleneck
        b = self.bottleneck(e4)  # [B, 256, H/8, W/8]

        # FPN: all outputs have base_ch*8 channels
        fpn_feats = self.fpn([b, e3, e2])  # [256,H/8], [256,H/4], [256,H/2]

        # Decoder with skip connections
        d3 = self.dec3(fpn_feats[0], fpn_feats[1])  # upsample fpn[0]+fpn[1] → [B, 128, H/4]
        d2 = self.dec2(d3, fpn_feats[2])             # upsample d3+fpn[2]    → [B, 64, H/2]
        d1 = self.dec1(d2, e1)                       # upsample d2+e1        → [B, 32, H, W]

        return self.out_conv(d1)  # [B, 2, H, W]

    def predict_confidence(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-pixel target confidence (softmax probability of class 1)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)[:, 1]  # [B, H, W]


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification.
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 ignore_index: int = -1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, C, H, W]
            targets: [B, H, W] long
        """
        B, C, H, W = logits.shape
        log_prob = F.log_softmax(logits, dim=1)
        prob = log_prob.exp()

        # Gather log_p for true class
        targets_flat = targets.view(-1)
        log_p = log_prob.permute(0, 2, 3, 1).reshape(-1, C)
        p = prob.permute(0, 2, 3, 1).reshape(-1, C)

        valid = targets_flat != self.ignore_index
        targets_valid = targets_flat[valid]
        log_p_valid = log_p[valid]
        p_valid = p[valid]

        log_pt = log_p_valid.gather(1, targets_valid.unsqueeze(1)).squeeze(1)
        pt = p_valid.gather(1, targets_valid.unsqueeze(1)).squeeze(1)

        # Alpha weighting
        alpha_t = torch.where(targets_valid == 1,
                              torch.tensor(self.alpha, device=logits.device),
                              torch.tensor(1 - self.alpha, device=logits.device))

        focal_weight = (1 - pt) ** self.gamma
        loss = -alpha_t * focal_weight * log_pt
        return loss.mean()


class RadarDataset(torch.utils.data.Dataset):
    """Dataset of 8-channel patches and their label maps."""

    def __init__(self, patches: 'np.ndarray', labels: 'np.ndarray',
                 augment: bool = True):
        import numpy as np
        self.patches = torch.from_numpy(patches.astype('float32'))
        self.labels = torch.from_numpy(labels.astype('int64'))
        self.augment = augment

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        x = self.patches[idx]
        y = self.labels[idx]
        if self.augment:
            x, y = self._augment(x, y)
        return x, y

    def _augment(self, x, y):
        """Random flip augmentations."""
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[2])
            y = torch.flip(y, dims=[1])
        if torch.rand(1) > 0.5:
            x = torch.flip(x, dims=[1])
            y = torch.flip(y, dims=[0])
        # SNR jitter on channel 6
        x[6] = x[6] + torch.randn_like(x[6]) * 0.05
        x = torch.clamp(x, 0, 1)
        return x, y
