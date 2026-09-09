import torch.nn as nn

class PromptProjector(nn.Module):
    """Map a global prompt vector to soft cross-attention tokens."""
    def __init__(self, t_dim, d_model, num_tokens=8):
        super().__init__()
        self.num_tokens = num_tokens
        self.lin = nn.Linear(t_dim, num_tokens * d_model)
    def forward(self, t_vec):
        x = self.lin(t_vec)
        B = x.size(0)
        return x.view(B, self.num_tokens, -1)
