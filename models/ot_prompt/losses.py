import torch
import torch.nn as nn
import torch.nn.functional as F

def pair_align_loss(hat_T, E_T, w_cos=1.0, w_l2=0.0):
    hat_T = torch.nan_to_num(hat_T.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    E_T = torch.nan_to_num(E_T.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    loss = 0.0
    if w_cos > 0:
        loss += w_cos * (1 - F.cosine_similarity(hat_T, E_T, dim=-1)).mean()
    if w_l2 > 0:
        loss += w_l2 * F.mse_loss(hat_T, E_T)
    return loss












def ot_cost_matrix_label_only(hat_T, E_T, labels, lambda_g=1.0):
    hat_T = torch.nan_to_num(hat_T.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    E_T = torch.nan_to_num(E_T.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    B, d = hat_T.shape
    dist_feat = torch.cdist(hat_T, E_T, p=2).pow(2).clamp_(0.0, 16.0)
    label_mask = (labels.unsqueeze(1) != labels.unsqueeze(0)).float()
    penalty = lambda_g * label_mask
    cost_matrix = dist_feat + penalty
    return cost_matrix


def sinkhorn_cost_from_matrix(cost_matrix, eps=0.5, iters=50):
    """Numerically stable balanced Sinkhorn cost in float32.

    The old implementation formed K=exp(-C/eps) and repeatedly divided by
    K@v.  That is fragile under AMP and can create Inf/NaN scaling vectors.
    This log-domain version keeps the transport plan finite even when a
    batch contains very dissimilar embeddings.
    """
    C = torch.nan_to_num(cost_matrix.float(), nan=0.0, posinf=16.0, neginf=0.0)
    C = C.clamp(0.0, 16.0)
    n, m = C.shape
    if n == 0 or m == 0:
        return C.new_zeros(())
    eps = max(float(eps), 1e-3)
    log_a = C.new_full((n,), -torch.log(C.new_tensor(float(n))))
    log_b = C.new_full((m,), -torch.log(C.new_tensor(float(m))))
    log_kernel = (-C / eps).clamp(-80.0, 0.0)
    log_u = C.new_zeros((n,))
    log_v = C.new_zeros((m,))
    for _ in range(max(1, int(iters))):
        log_u = log_a - torch.logsumexp(log_kernel + log_v.unsqueeze(0), dim=1)
        log_v = log_b - torch.logsumexp(log_kernel.transpose(0, 1) + log_u.unsqueeze(0), dim=1)
    log_plan = (log_u.unsqueeze(1) + log_kernel + log_v.unsqueeze(0)).clamp(-80.0, 10.0)
    plan = torch.exp(log_plan)
    loss = (plan * C).sum()
    return torch.nan_to_num(loss, nan=0.0, posinf=16.0, neginf=0.0).clamp_min(0.0)


def mmd_rbf(X, Y, sigma=1.0):
    X = torch.nan_to_num(X.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    Y = torch.nan_to_num(Y.float(), nan=0.0, posinf=1.0, neginf=-1.0)
    XX = torch.cdist(X, X, p=2).pow(2).clamp_(0.0, 16.0)
    YY = torch.cdist(Y, Y, p=2).pow(2).clamp_(0.0, 16.0)
    XY = torch.cdist(X, Y, p=2).pow(2).clamp_(0.0, 16.0)
    Kxx = torch.exp(-XX/(2*sigma*sigma))
    Kyy = torch.exp(-YY/(2*sigma*sigma))
    Kxy = torch.exp(-XY/(2*sigma*sigma))

    mmd = Kxx.mean() + Kyy.mean() - 2*Kxy.mean()
    return torch.nan_to_num(mmd, nan=0.0, posinf=1.0, neginf=0.0).clamp_min(0.0)

def temporal_tv(T_seq, flow_mask=None):
    if T_seq.dim() == 2:
        return T_seq.new_zeros(())

    diff = T_seq[:, 1:, :] - T_seq[:, :-1, :]
    if flow_mask is not None:
        diff = diff * (1.0 - flow_mask)
    return diff.abs().mean()


def grade_center_contrast_loss(hat_T, centers, labels, temperature=0.2):
    """Class-center contrastive loss for pseudo-text embeddings."""
    if hat_T is None or centers is None or labels is None:
        return hat_T.new_zeros(()) if isinstance(hat_T, torch.Tensor) else torch.tensor(0.0)
    labels = labels.long().view(-1)
    hat_T = F.normalize(hat_T, dim=-1)
    centers = F.normalize(centers, dim=-1)
    logits = hat_T @ centers.t()
    logits = logits / max(float(temperature), 1e-6)
    return F.cross_entropy(logits, labels)


def bidirectional_info_nce_loss(hat_T, E_T, temperature=0.07):
    """Paired cross-modal InfoNCE between pseudo-text and real-text embeddings.

    This is the standard contrastive baseline for OT-vs-contrastive ablation:
    each visual pseudo-text embedding is pulled to its paired clinical text
    embedding while other batch samples are treated as negatives.
    """
    if hat_T is None or E_T is None:
        return hat_T.new_zeros(()) if isinstance(hat_T, torch.Tensor) else torch.tensor(0.0)
    if hat_T.size(0) <= 1:
        return pair_align_loss(hat_T, E_T, w_cos=1.0, w_l2=0.0)
    hat_T = F.normalize(hat_T, dim=-1)
    E_T = F.normalize(E_T, dim=-1)
    logits = hat_T @ E_T.t()
    logits = logits / max(float(temperature), 1e-6)
    targets = torch.arange(logits.size(0), device=logits.device)
    loss_h2t = F.cross_entropy(logits, targets)
    loss_t2h = F.cross_entropy(logits.t(), targets)
    return 0.5 * (loss_h2t + loss_t2h)


def supervised_cross_modal_contrastive_loss(hat_T, E_T, labels, temperature=0.07):
    """Label-aware cross-modal supervised contrastive alignment.

    Positives are all real-text embeddings in the current mini-batch with the
    same BI-RADS label; negatives are embeddings from different labels. The loss
    is computed symmetrically from pseudo->real and real->pseudo.
    """
    if hat_T is None or E_T is None or labels is None:
        return hat_T.new_zeros(()) if isinstance(hat_T, torch.Tensor) else torch.tensor(0.0)
    labels = labels.long().view(-1)
    if hat_T.size(0) <= 1:
        return pair_align_loss(hat_T, E_T, w_cos=1.0, w_l2=0.0)

    hat_T = F.normalize(hat_T, dim=-1)
    E_T = F.normalize(E_T, dim=-1)
    logits = (hat_T @ E_T.t()) / max(float(temperature), 1e-6)
    pos = labels.unsqueeze(1).eq(labels.unsqueeze(0)).float()

    def directional_loss(x):
        log_prob = x - torch.logsumexp(x, dim=1, keepdim=True)
        pos_count = pos.sum(dim=1).clamp_min(1.0)
        return -((pos * log_prob).sum(dim=1) / pos_count).mean()

    loss_h2t = directional_loss(logits)
    loss_t2h = directional_loss(logits.t())
    return 0.5 * (loss_h2t + loss_t2h)
