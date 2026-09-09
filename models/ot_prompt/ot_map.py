import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalPool(nn.Module):
    """Simple temporal pooling (can be replaced by TransformerPooling)"""
    def __init__(self, mode='mean'):
        super().__init__()
        assert mode in ['mean', 'max']
        self.mode = mode

    def forward(self, v):
        if v.dim() == 3:
            if self.mode == 'mean':
                return v.mean(dim=1, keepdim=True)
            else:
                return v.max(dim=1).values
        else:
            raise ValueError("Input v must have 3 dimensions [B, T, C]")

class OTMap(nn.Module):
    """
    Monge mapping T_theta: V -> T, learns to map video features to text embedding space.
    in_dim: dimension of video feature
    t_dim: dimension of text embedding (matches the text encoder output, like 256/512)
    num_classes: number of label categories (e.g., 4 for L2-L5)
    """
    def __init__(
        self,
        in_dim,
        t_dim,
        num_classes=4,
        hidden=512,
        pool='mean',
        strong_label_prompt=False,
        label_residual_min_scale=0.0,
    ):
        super().__init__()
        self.pool = TemporalPool(mode=pool)
        self.in_dim = in_dim
        self.t_dim = t_dim
        self.strong_label_prompt = bool(strong_label_prompt)
        self.label_residual_min_scale = float(label_residual_min_scale)
        self.label_embed = nn.Embedding(num_classes, in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, t_dim)
        )
        self.label_residual = nn.Linear(in_dim, t_dim, bias=False)
        if in_dim == t_dim:
            nn.init.eye_(self.label_residual.weight)
        else:
            nn.init.xavier_uniform_(self.label_residual.weight)
        init_extra_scale = 0.02
        if self.strong_label_prompt:
            init_raw = torch.log(torch.expm1(torch.tensor(init_extra_scale)))
            self.label_residual_scale = nn.Parameter(init_raw.clone().detach())
        else:
            self.label_residual_scale = nn.Parameter(torch.tensor(init_extra_scale))

    def _label_scale(self):
        if self.strong_label_prompt:
            raw = self.label_residual_scale
            extra = torch.where(raw > -1.0, raw.clamp(min=0.0, max=0.05), F.softplus(raw))
            return self.label_residual_min_scale + extra
        return self.label_residual_scale

    def map_vectors(self, v, labels=None):
        if labels is None:
            label_emb = torch.zeros(v.shape[0], self.in_dim, device=v.device, dtype=v.dtype)
            residual = torch.zeros(v.shape[0], self.t_dim, device=v.device, dtype=v.dtype)
        else:
            label_emb = self.label_embed(labels).to(dtype=v.dtype)
            residual = self.label_residual(label_emb)
        v_cat = torch.cat([v, label_emb], dim=-1)
        return self.mlp(v_cat) + self._label_scale() * residual

    def forward(self, v_seq, labels=None):
        if v_seq.dim() == 5:
            B, C, H, W, T = v_seq.shape
            v_seq = v_seq.flatten(2, 3).permute(0, 4, 1, 2).flatten(2).mean(-1)
        elif v_seq.dim() == 4:
            v_seq = v_seq.mean(dim=[2, 3]).unsqueeze(1)

        if v_seq.dim() == 3:
            if torch.isnan(v_seq).any() or torch.isinf(v_seq).any():
                print(f"NaN or Inf detected in v_seq before pooling! Shape: {v_seq.shape}")
                v_seq = torch.nan_to_num(v_seq, nan=0.0, posinf=1e10, neginf=-1e10)
            v = self.pool(v_seq)
        else:
            raise ValueError("v_seq should have 3 dimensions [B, T, C], found: {}".format(v_seq.shape))

        v = v.squeeze(1)
        v = v / (v.abs().max() + 1e-8)

        return self.map_vectors(v, labels=labels)
