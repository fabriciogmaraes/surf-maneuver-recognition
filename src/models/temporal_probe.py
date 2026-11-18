"""Small classifiers on frozen V-JEPA 2 features of shape (time, dim)."""
import torch
from torch import nn


class TemporalProbe(nn.Module):
    """Classify a clip from its sequence of time-slice features.

    Each time slice is projected to a smaller size, the slices exchange information through
    one self-attention layer, and a learned query pools them into a single vector for the classifier.
    With `use_order=False` there are no positional embeddings, so the model cannot tell the
    slices' order apart: the same architecture, blind to time order (the ablation baseline).
    """

    def __init__(self, in_dim=1024, hidden=256, num_classes=4, num_slices=8,
                 heads=4, dropout=0.3, use_order=True):
        super().__init__()
        self.proj = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Dropout(dropout)
        )
        self.pos = nn.Parameter(torch.zeros(1, num_slices, hidden)) if use_order else None
        self.mixer = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=2 * hidden, dropout=dropout,
            batch_first=True, norm_first=True,
        )
        self.query = nn.Parameter(torch.zeros(1, 1, hidden))
        self.pool = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, num_classes))

        nn.init.trunc_normal_(self.query, std=0.02)
        if self.pos is not None:
            nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):                      # x: (batch, time, in_dim)
        h = self.proj(x)                       # (batch, time, hidden)
        if self.pos is not None:
            h = h + self.pos                   # tag each slice with its position in time
        h = self.mixer(h)                      # slices exchange information
        query = self.query.expand(len(h), -1, -1)
        pooled, _ = self.pool(query, h, h)     # (batch, 1, hidden)
        return self.head(pooled.squeeze(1))    # (batch, num_classes)