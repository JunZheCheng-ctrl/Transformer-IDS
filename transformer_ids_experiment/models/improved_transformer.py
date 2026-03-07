import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureGate(nn.Module):
    """
    Feature-level attention / gating:
    x: [B, D]
    output: [B, D]
    """
    def __init__(self, d_in: int, hidden: int = 128, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_in),
            nn.Sigmoid(),
        )

    def forward(self, x):
        g = self.net(x)
        return x * g


class TabTransformerEncoder(nn.Module):
    """
    Tabular Transformer over "feature tokens":
    Convert [B, D] -> tokens [B, D, 1] -> embed -> Transformer -> pool -> vector
    """
    def __init__(
            self,
            d_in: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_model = d_model

        # each feature is a token with 1 scalar -> embed to d_model
        self.token_embed = nn.Linear(1, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        # x: [B, D]
        x = x.unsqueeze(-1)             # [B, D, 1]
        tok = self.token_embed(x)       # [B, D, d_model]
        tok = self.encoder(tok)         # [B, D, d_model]
        tok = self.norm(tok)
        # mean pool across tokens (features)
        z = tok.mean(dim=1)             # [B, d_model]
        return z


class TemperatureScaler(nn.Module):
    """
    Calibration layer (temperature scaling).
    Use in eval-time or fine-tune on val set.
    """
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_t = nn.Parameter(torch.log(torch.tensor(init_temp)))

    def forward(self, logits):
        t = torch.exp(self.log_t).clamp(0.5, 10.0)
        return logits / t


class ImprovedTransformerIDS(nn.Module):
    """
    IFT = FeatureGate + TabTransformerEncoder + classifier + optional temperature scaling
    Output logits (before sigmoid)
    """
    def __init__(
            self,
            d_in: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            ff: int = 128,
            dropout: float = 0.1,
            use_calibration: bool = True,
    ):
        super().__init__()
        self.gate = FeatureGate(d_in, hidden=min(128, max(32, d_in)), dropout=dropout)
        self.encoder = TabTransformerEncoder(
            d_in=d_in,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=ff,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )
        self.calibrator = TemperatureScaler(1.0) if use_calibration else None

    def forward(self, x):
        # x: [B, D]
        x = self.gate(x)
        z = self.encoder(x)
        logits = self.classifier(z).squeeze(-1)  # [B]
        if self.calibrator is not None:
            logits = self.calibrator(logits)
        return logits