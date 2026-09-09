import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _safe_gn(num_channels: int, gn_groups: int) -> nn.GroupNorm:
    if gn_groups <= 1 or gn_groups >= num_channels:
        return nn.GroupNorm(1, num_channels)
    g = math.gcd(num_channels, gn_groups)
    if g == 0:
        g = 1
    return nn.GroupNorm(g, num_channels)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution with dilation-aware padding."""
    def __init__(self, in_ch, out_ch, k=3, s=1, dilation=1, gn_groups=32):
        super().__init__()
        p = dilation * (k - 1) // 2
        self.dw = nn.Conv2d(in_ch, in_ch, k, stride=s, padding=p, dilation=dilation,
                            groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.gn = _safe_gn(out_ch, gn_groups)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.gn(x)
        x = self.act(x)
        return x


class TinyEncoder(nn.Module):
    """Lightweight feature extractor with optional dilated convolution."""
    def __init__(self, in_ch=3, feat_ch=256, gn_groups=32, use_dilation=True):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1, bias=False),
            _safe_gn(64, gn_groups),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, feat_ch, 3, padding=1, bias=False),
            _safe_gn(feat_ch, gn_groups),
            nn.ReLU(inplace=True),
        )
        if use_dilation:
            self.block3 = nn.Sequential(
                nn.Conv2d(feat_ch, feat_ch, 3, padding=2, dilation=2, bias=False),
                _safe_gn(feat_ch, gn_groups),
                nn.ReLU(inplace=True),
            )
        else:
            self.block3 = nn.Identity()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x


class ConvGRUCell(nn.Module):
    def __init__(self, in_ch, hidden_ch, k=3):
        super().__init__()
        p = k // 2
        self.hidden_ch = hidden_ch
        self.conv_zr = nn.Conv2d(in_ch + hidden_ch, 2 * hidden_ch, k, padding=p, bias=True)
        self.conv_h  = nn.Conv2d(in_ch + hidden_ch, hidden_ch, k, padding=p, bias=True)

    def forward(self, x, h=None):
        if h is None:
            h = torch.zeros(x.size(0), self.hidden_ch, x.size(2), x.size(3),
                            device=x.device, dtype=x.dtype)
        xh = torch.cat([x, h], dim=1)
        zr = torch.sigmoid(self.conv_zr(xh))
        z, r = torch.chunk(zr, 2, dim=1)
        h_tilde = torch.tanh(self.conv_h(torch.cat([x, r * h], dim=1)))
        h_new = (1 - z) * h + z * h_tilde
        return h_new


class SmallASPP(nn.Module):
    """Compact ASPP block with parallel dilated and pooled branches."""
    def __init__(self, in_ch, out_ch, rates=(1, 3, 5), use_depthwise=False, gn_groups=32):
        super().__init__()
        self.branches = nn.ModuleList()
        for r in rates:
            if use_depthwise:
                self.branches.append(
                    DepthwiseSeparableConv(in_ch, out_ch, k=3, s=1, dilation=r, gn_groups=gn_groups)
                )
            else:
                self.branches.append(
                    nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, 3, padding=r, dilation=r, bias=False),
                        _safe_gn(out_ch, gn_groups),
                        nn.ReLU(inplace=True),
                    )
                )
        self.branch_1x1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            _safe_gn(out_ch, gn_groups),
            nn.ReLU(inplace=True),
        )
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            _safe_gn(out_ch, gn_groups),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False),
            _safe_gn(out_ch, gn_groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        feats.append(self.branch_1x1(x))
        gp = self.global_pool(x)
        gp = F.interpolate(gp, size=x.shape[-2:], mode='bilinear', align_corners=False)
        feats.append(gp)
        x = torch.cat(feats, dim=1)
        x = self.fuse(x)
        return x


class EnhancedDecoder(nn.Module):
    """
    Advanced decoder with optional ASPP / depthwise / skip
    """
    def __init__(self, in_ch, skip_ch=0, mid_ch=128, out_ch=1,
                 use_aspp=False, use_depthwise=False, use_skip=False,
                 return_logits=True, gn_groups=32):
        super().__init__()
        self.return_logits = return_logits
        self.use_skip = use_skip
        dec_in = in_ch + (skip_ch if use_skip and skip_ch > 0 else 0)

        if use_aspp:
            self.pre = SmallASPP(dec_in, mid_ch, rates=(1, 3, 5), use_depthwise=use_depthwise, gn_groups=gn_groups)
            block_in = mid_ch
        else:
            self.pre = nn.Identity()
            block_in = dec_in

        if use_depthwise:
            self.block = nn.Sequential(
                DepthwiseSeparableConv(block_in, mid_ch, k=3, gn_groups=gn_groups),
                DepthwiseSeparableConv(mid_ch, mid_ch, k=3, gn_groups=gn_groups),
            )
        else:
            self.block = nn.Sequential(
                nn.Conv2d(block_in, mid_ch, 3, padding=1, bias=False),
                _safe_gn(mid_ch, gn_groups),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
                _safe_gn(mid_ch, gn_groups),
                nn.ReLU(inplace=True),
            )
        self.head = nn.Conv2d(mid_ch, out_ch, 1)

    def forward(self, h, skip_feat=None):
        x = h if (not self.use_skip or skip_feat is None) else torch.cat([h, skip_feat], dim=1)
        x = self.pre(x)
        x = self.block(x)
        logits = self.head(x)
        return logits if self.return_logits else torch.sigmoid(logits)


class TemporalBranch(nn.Module):
    """Temporal propagation branch with mask-guided recurrent decoding."""
    def __init__(self,
                 feat_ch=256,
                 hidden_ch=256,
                 use_exist_head=False,
                 shared_encoder: nn.Module = None,
                 freeze_encoder=True,
                 return_logits=True,
                 iterative_guidance=True,
                 gn_groups=32,
                 learnable_alpha=False,
                 alpha_channelwise=False,
                 alpha_init=1.0,
                 use_guide_blend=False,
                 learnable_beta=True,
                 beta_init=0.5,
                 use_decoder_aspp=False,
                 use_decoder_depthwise=False,
                 use_decoder_skip=False,
                 dual_scale=False,
                 hidden_ch_low=None,
                 gamma_init=0.5,
                 decode_head_first: bool = False,
                 learnable_gate=True,
                 gate_init=0.5,
                 ):
        super().__init__()
        self.encoder = shared_encoder if shared_encoder is not None else TinyEncoder(in_ch=3, feat_ch=feat_ch, gn_groups=gn_groups)
        if freeze_encoder and shared_encoder is not None:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        self.mask_embed = nn.Sequential(
            nn.Conv2d(1, feat_ch, 1, bias=True),
            _safe_gn(feat_ch, gn_groups),
            nn.ReLU(inplace=True),
        )
        try:
            conv0 = self.mask_embed[0]
            if isinstance(conv0, nn.Conv2d) and conv0.bias is not None:
                nn.init.constant_(conv0.bias, 0.0)
        except Exception:
            pass
        if learnable_gate:
            self.gate = nn.Parameter(torch.tensor(gate_init, dtype=torch.float32))
        else:
            self.gate = torch.tensor(gate_init, dtype=torch.float32)
        self.fuse = nn.Sequential(
            nn.Conv2d(feat_ch * 2, feat_ch, 1, bias=False),
            _safe_gn(feat_ch, gn_groups),
            nn.ReLU(inplace=True),
        )

        self.h_init = nn.Sequential(
            nn.Conv2d(feat_ch, hidden_ch, 1, bias=True),
            _safe_gn(hidden_ch, gn_groups),
        )

        self.learnable_alpha = learnable_alpha
        self.alpha_channelwise = alpha_channelwise
        if learnable_alpha:
            if alpha_channelwise:
                self.alpha = nn.Parameter(torch.full((1, feat_ch, 1, 1), float(alpha_init)))
            else:
                self.alpha = nn.Parameter(torch.tensor(float(alpha_init)))
        else:
            self.register_buffer('alpha', torch.tensor(float(alpha_init)), persistent=False)

        self.use_guide_blend = use_guide_blend
        self.learnable_beta = learnable_beta
        if use_guide_blend:
            if learnable_beta:
                self.beta_raw = nn.Parameter(torch.tensor(float(beta_init)))
            else:
                self.register_buffer('beta_raw', torch.tensor(float(beta_init)), persistent=False)

        self.gru_high = ConvGRUCell(in_ch=feat_ch, hidden_ch=hidden_ch)

        self.dual_scale = dual_scale
        if dual_scale:
            self.pool2 = nn.AvgPool2d(2)
            self.gru_low = ConvGRUCell(in_ch=feat_ch, hidden_ch=(hidden_ch_low or hidden_ch))
            self.h_init_low = nn.Conv2d(feat_ch, (hidden_ch_low or hidden_ch), 1)
            self.proj_low2high = nn.Conv2d((hidden_ch_low or hidden_ch), hidden_ch, 1)
            self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        self.return_logits = return_logits
        self.use_decoder_skip = use_decoder_skip
        self.decoder = EnhancedDecoder(
            in_ch=hidden_ch,
            skip_ch=(feat_ch if use_decoder_skip else 0),
            mid_ch=128,
            out_ch=1,
            use_aspp=use_decoder_aspp,
            use_depthwise=use_decoder_depthwise,
            use_skip=use_decoder_skip,
            return_logits=return_logits,
            gn_groups=gn_groups
        )

        self.iterative_guidance = iterative_guidance

        self.use_exist_head = use_exist_head
        if use_exist_head:
            self.exist_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(hidden_ch, 1, 1))

        self.decode_head_first = bool(decode_head_first)

    def _normalize_mask_tensor(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Ensure mask is float in [0,1]. Accepts 0/255 or 0/1 inputs.
        """
        if not torch.is_floating_point(mask):
            mask = mask.float()
        if mask.max() > 1.0:
            mask = mask / 255.0
        return mask.clamp(0.0, 1.0)

    @staticmethod
    def corrupt_mask_for_training(mask: torch.Tensor, prob_erase: float = 0.2, max_erode: int = 3) -> torch.Tensor:
        """
        Simple mask corruption for training robustness.
        Input/Output: tensor B,1,H,W or 1,H,W - values in {0,1} or floats
        """
        m = mask.clone()
        if not torch.is_floating_point(m):
            m = m.float()
        if m.max() > 1.0:
            m = (m / 255.0).clamp(0.0, 1.0)

        B = m.shape[0]
        H, W = m.shape[-2], m.shape[-1]
        for b in range(B):
            if torch.rand(1).item() < prob_erase:
                ys = torch.randint(0, H, (2,))
                xs = torch.randint(0, W, (2,))
                y0, y1 = int(min(ys)), int(max(ys))
                x0, x1 = int(min(xs)), int(max(xs))
                y1 = min(H, y0 + torch.randint(1, max(2, max_erode), (1,)).item())
                x1 = min(W, x0 + torch.randint(1, max(2, max_erode), (1,)).item())
                m[b, :, y0:y1, x0:x1] = 0.0
        return m

    def _alpha(self):
        a = self.alpha
        if isinstance(a, nn.Parameter) or (isinstance(a, torch.Tensor) and a.requires_grad):
            return torch.sigmoid(a)
        return a

    def _beta(self):
        if not self.use_guide_blend:
            return None
        b = self.beta_raw
        return torch.sigmoid(b) if isinstance(b, nn.Parameter) else b.clamp(0.0, 1.0)

    def _build_guide(self, img_feat_for_guide: torch.Tensor, mask_prob: torch.Tensor, per_channel_norm: bool = False) -> torch.Tensor:
        """
        Build guide robustly:
         - normalize mask to [0,1]
         - compute mask embedding via mask_embed (Conv->GN->ReLU)
         - optionally per-channel normalize mask embedding
         - fuse with img_feat (Conv->GN->ReLU)
        """
        mp = self._normalize_mask_tensor(mask_prob)
        m_emb = self.mask_embed(mp)
        if per_channel_norm:
            mean = m_emb.mean(dim=[2,3], keepdim=True)
            std = m_emb.std(dim=[2,3], keepdim=True) + 1e-6
            m_emb = (m_emb - mean) / std
        cat = torch.cat([img_feat_for_guide, m_emb], dim=1)
        raw_guide = self.fuse(cat)
        return raw_guide

    def _step_update(self, feat_t, guide_t, h_high, h_low=None):
        a = self._alpha()
        if isinstance(a, torch.Tensor) and a.ndim == 4:
            x_high = feat_t + a * guide_t
        else:
            x_high = feat_t + float(a.item() if isinstance(a, torch.Tensor) else a) * guide_t

        if self.dual_scale and (h_low is not None):
            up = F.interpolate(self.proj_low2high(h_low), size=feat_t.shape[-2:], mode='bilinear', align_corners=False)
            x_high = x_high + self.gamma * up

        h_high = self.gru_high(x_high, h_high)
        return h_high

    def _forward_segment_legacy(self, mem_img, mem_mask, frames, valid_time_mask=None,
                        last_only: bool = True, detach_bptt: bool = True,
                        per_channel_norm_guide: bool = False,
                        return_endpoints: bool = False):
        """Run recurrent propagation from a memory frame to a dynamic clip."""
        B, Tp = frames.shape[:2]
        if Tp == 0:
            if last_only:
                return mem_mask, None
            empty = mem_mask.new_zeros((B, 0, 1, mem_mask.shape[-2], mem_mask.shape[-1]))
            return empty, None

        mem_mask = self._normalize_mask_tensor(mem_mask)
        mem_feat_high = self.encoder(mem_img)
        prev_mask = mem_mask

        def blend_for_guide(feat_high_t):
            if not self.use_guide_blend:
                return mem_feat_high
            beta = self._beta()
            return beta * mem_feat_high + (1.0 - beta) * feat_high_t

        guide0_high = self._build_guide(blend_for_guide(mem_feat_high), prev_mask, per_channel_norm=per_channel_norm_guide)
        h_high = self.h_init(guide0_high)

        if self.dual_scale:
            mem_feat_low = F.avg_pool2d(mem_feat_high, 2)
            guide0_low = F.avg_pool2d(guide0_high, 2)
            h_low = self.h_init_low(guide0_low)
        else:
            h_low = None

        def decode(h_cur, feat_for_skip=None):
            out = self.decoder(h_cur, feat_for_skip if self.use_decoder_skip else None)
            return out

        if last_only:
            for t in range(Tp - 1):
                with torch.no_grad():
                    feat_t_high = self.encoder(frames[:, t])
                    guide_t_high = self._build_guide(blend_for_guide(feat_t_high) if self.use_guide_blend else blend_for_guide(feat_t_high),
                                                     prev_mask, per_channel_norm=per_channel_norm_guide)
                    if self.dual_scale:
                        feat_t_low = F.avg_pool2d(feat_t_high, 2)
                        guide_t_low = F.avg_pool2d(guide_t_high, 2)
                        x_low = feat_t_low + self._alpha() * guide_t_low
                        h_low = self.gru_low(x_low, h_low)

                    h_high = self._step_update(feat_t_high, guide_t_high, h_high, h_low)

                    if self.iterative_guidance:
                        tmp_logits = decode(h_high, feat_t_high if self.use_decoder_skip else None)
                        tmp_prob = torch.sigmoid(tmp_logits) if self.return_logits else tmp_logits
                        prev_mask = tmp_prob.clamp(0, 1)

                if detach_bptt and (h_high is not None):
                    h_high = h_high.detach()
                    if self.dual_scale and (h_low is not None):
                        h_low = h_low.detach()

            feat_last_high = self.encoder(frames[:, Tp - 1])
            guide_last_high = self._build_guide(blend_for_guide(feat_last_high) if self.use_guide_blend else blend_for_guide(feat_last_high),
                                                prev_mask, per_channel_norm=per_channel_norm_guide)

            if self.dual_scale:
                feat_last_low = F.avg_pool2d(feat_last_high, 2)
                guide_last_low = F.avg_pool2d(guide_last_high, 2)
                x_low = feat_last_low + self._alpha() * guide_last_low
                h_low = self.gru_low(x_low, h_low)

            h_high = self._step_update(feat_last_high, guide_last_high, h_high, h_low)
            pred_last = decode(h_high, feat_last_high if self.use_decoder_skip else None)

            if self.use_exist_head:
                exist_last = torch.sigmoid(self.exist_head(h_high)).flatten(1)
            else:
                exist_last = None

            return pred_last, exist_last





        preds = []
        exists = [] if self.use_exist_head else None
        for t in range(Tp):
            if t == 0 or t == Tp-1:
                feat_t_high = self.encoder(frames[:, t])
                guide_t_high = self._build_guide(blend_for_guide(feat_t_high) if self.use_guide_blend else blend_for_guide(feat_t_high),
                                                prev_mask, per_channel_norm=per_channel_norm_guide)

                if self.dual_scale:
                    feat_t_low = F.avg_pool2d(feat_t_high, 2)
                    guide_t_low = F.avg_pool2d(guide_t_high, 2)
                    x_low = feat_t_low + self._alpha() * guide_t_low
                    h_low = self.gru_low(x_low, h_low)

                h_high = self._step_update(feat_t_high, guide_t_high, h_high, h_low)
                out = decode(h_high, feat_t_high if self.use_decoder_skip else None)
                if (not return_endpoints) or t in (0, Tp - 1):
                    preds.append(out)
                if self.iterative_guidance:
                    prob_t = torch.sigmoid(out) if self.return_logits else out
                    prev_mask = prob_t.clamp(0, 1)
                if self.use_exist_head:
                    e = torch.sigmoid(self.exist_head(h_high)).flatten(1)
                    exists.append(e)
            else:
                with torch.no_grad():
                    feat_t_high = self.encoder(frames[:, t])
                    guide_t_high = self._build_guide(blend_for_guide(feat_t_high) if self.use_guide_blend else blend_for_guide(feat_t_high),
                                                    prev_mask, per_channel_norm=per_channel_norm_guide)

                    if self.dual_scale:
                        feat_t_low = F.avg_pool2d(feat_t_high, 2)
                        guide_t_low = F.avg_pool2d(guide_t_high, 2)
                        x_low = feat_t_low + self._alpha() * guide_t_low
                        h_low = self.gru_low(x_low, h_low)

                    h_high = self._step_update(feat_t_high, guide_t_high, h_high, h_low)
                    out = decode(h_high, feat_t_high if self.use_decoder_skip else None)
                    if not return_endpoints:
                        preds.append(out)
                    if self.iterative_guidance:
                        prob_t = torch.sigmoid(out) if self.return_logits else out
                        prev_mask = prob_t.clamp(0, 1)
                    if self.use_exist_head:
                        e = torch.sigmoid(self.exist_head(h_high)).flatten(1)
                        exists.append(e)
                h_high = h_high.detach()
                if self.dual_scale and (h_low is not None):
                    h_low = h_low.detach()





        preds = torch.stack(preds, dim=1)
        if self.use_exist_head:
            exists = torch.stack(exists, dim=1).unsqueeze(-1)
        else:
            exists = None
        return preds, exists

    def _init_state(self, mem_img, mem_mask, per_channel_norm_guide=False):
        """Initialize recurrent state from the actual segment-start frame."""
        mem_mask = self._normalize_mask_tensor(mem_mask)
        mem_feat_high = self.encoder(mem_img)
        guide0_high = self._build_guide(
            mem_feat_high, mem_mask, per_channel_norm=per_channel_norm_guide
        )
        h_high = self.h_init(guide0_high)
        h_low = self.h_init_low(F.avg_pool2d(guide0_high, 2)) if self.dual_scale else None
        return mem_mask, mem_feat_high, h_high, h_low

    def _decode_state(self, h_high, feat_high):
        return self.decoder(h_high, feat_high if self.use_decoder_skip else None)

    def forward_initial(self, mem_img, mem_mask, per_channel_norm_guide=False):
        """Decode the segment-start prediction without an extra frame update."""
        _, mem_feat_high, h_high, _ = self._init_state(
            mem_img, mem_mask, per_channel_norm_guide=per_channel_norm_guide
        )
        return self._decode_state(h_high, mem_feat_high)

    def forward_segment(self, mem_img, mem_mask, frames, valid_time_mask=None,
                        last_only: bool = True, detach_bptt: bool = True,
                        per_channel_norm_guide: bool = False,
                        return_endpoints: bool = False):
        """Train a variable-length segment with correctly aligned endpoints.

        ``mem_img``/``mem_mask`` are the segment-start frame and ``frames``
        excludes that frame. The first returned endpoint is the direct
        ``h_init`` prediction for the segment start; the second is the last
        valid temporal frame. Padded frames are never processed as real data.
        """
        B, Tp = frames.shape[:2]
        if valid_time_mask is None:
            lengths = [Tp] * B
        else:
            lengths = valid_time_mask.to(device=frames.device, dtype=torch.bool).sum(dim=1).tolist()

        start_preds, end_preds, exist_preds = [], [], []
        for b, length_value in enumerate(lengths):
            length = max(0, min(int(length_value), Tp))
            mem_b = mem_img[b:b + 1]
            mask_b = mem_mask[b:b + 1]
            prev_mask, mem_feat_high, h_high, h_low = self._init_state(
                mem_b, mask_b, per_channel_norm_guide=per_channel_norm_guide
            )
            start_out = self._decode_state(h_high, mem_feat_high)
            start_preds.append(start_out)
            end_out = start_out
            end_exist = None

            for t in range(length):
                is_last = (t == length - 1)
                context = torch.enable_grad() if (is_last or not detach_bptt) else torch.no_grad()
                with context:
                    feat_t_high = self.encoder(frames[b:b + 1, t])
                    if self.use_guide_blend:
                        feat_for_guide = self._beta() * mem_feat_high + (1.0 - self._beta()) * feat_t_high
                    else:
                        feat_for_guide = mem_feat_high
                    guide_t = self._build_guide(
                        feat_for_guide, prev_mask,
                        per_channel_norm=per_channel_norm_guide
                    )
                    if self.dual_scale:
                        feat_t_low = F.avg_pool2d(feat_t_high, 2)
                        guide_t_low = F.avg_pool2d(guide_t, 2)
                        h_low = self.gru_low(
                            feat_t_low + self._alpha() * guide_t_low, h_low
                        )
                    h_high = self._step_update(
                        feat_t_high, guide_t, h_high,
                        h_low if self.dual_scale else None
                    )
                    out = self._decode_state(h_high, feat_t_high)
                    if self.iterative_guidance:
                        prev_mask = torch.sigmoid(out) if self.return_logits else out
                        prev_mask = prev_mask.clamp(0, 1)

                if is_last:
                    end_out = out
                    if self.use_exist_head:
                        end_exist = torch.sigmoid(self.exist_head(h_high)).flatten(1)
                elif detach_bptt:
                    h_high = h_high.detach()
                    if self.dual_scale and h_low is not None:
                        h_low = h_low.detach()

            end_preds.append(end_out)
            if self.use_exist_head:
                exist_preds.append(end_exist if end_exist is not None else torch.zeros(
                    (1, 1), device=mem_img.device, dtype=mem_img.dtype
                ))

        start_preds = torch.cat(start_preds, dim=0)
        end_preds = torch.cat(end_preds, dim=0)
        if last_only:
            exists = torch.cat(exist_preds, dim=0) if self.use_exist_head else None
            return end_preds, exists
        preds = torch.stack([start_preds, end_preds], dim=1)
        exists = torch.stack(exist_preds, dim=1).unsqueeze(-1) if self.use_exist_head else None
        return preds, exists

    @torch.no_grad()
    def forward_roll(self, mem_img, mem_mask, frames, per_channel_norm_guide: bool = False, decode_head_first: bool = None):
        """Roll the branch through a clip and return probability masks."""
        self.eval()
        B, Tp = frames.shape[:2]
        if Tp == 0:
            return mem_mask.new_zeros((B, 0, 1, mem_mask.shape[-2], mem_mask.shape[-1]))

        mem_mask = self._normalize_mask_tensor(mem_mask)

        mem_feat_high = self.encoder(mem_img)
        prev_mask = mem_mask

        def blend_for_guide(feat_high_t):
            if not self.use_guide_blend:
                return mem_feat_high
            beta = self._beta()
            return beta * mem_feat_high + (1.0 - beta) * feat_high_t

        guide0_high = self._build_guide(blend_for_guide(mem_feat_high), prev_mask, per_channel_norm=per_channel_norm_guide)
        h_high = self.h_init(guide0_high)

        if self.dual_scale:
            mem_feat_low = F.avg_pool2d(mem_feat_high, 2)
            guide0_low = F.avg_pool2d(guide0_high, 2)
            h_low = self.h_init_low(guide0_low)
        else:
            h_low = None

        preds = []
        decode_first_direct = self.decode_head_first if decode_head_first is None else bool(decode_head_first)
        for t in range(Tp):
            feat_t_high = self.encoder(frames[:, t])
            guide_t_high = self._build_guide(blend_for_guide(feat_t_high) if self.use_guide_blend else blend_for_guide(feat_t_high),
                                             prev_mask, per_channel_norm=per_channel_norm_guide)

            if self.dual_scale:
                feat_t_low = F.avg_pool2d(feat_t_high, 2)
                guide_t_low = F.avg_pool2d(guide_t_high, 2)
                x_low = feat_t_low + self._alpha() * guide_t_low
                h_low = self.gru_low(x_low, h_low)

            h_high = self._step_update(feat_t_high, guide_t_high, h_high, h_low)

            if t == 0 and decode_first_direct:
                out = self.decoder(h_high, feat_t_high if self.use_decoder_skip else None)
            else:
                out = self.decoder(h_high, feat_t_high if self.use_decoder_skip else None)

            prob = torch.sigmoid(out) if self.return_logits else out
            preds.append(prob.clamp(0, 1))
            if self.iterative_guidance:
                prev_mask = preds[-1]
        preds = torch.stack(preds, dim=1)
        return preds
