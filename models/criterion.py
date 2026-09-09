import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from misc import (
    nested_tensor_from_tensor_list,
    get_world_size, is_dist_avail_and_initialized
)

from models.GroundingDINO.segmentation import (dice_loss, sigmoid_focal_loss)
from einops import rearrange


def sanitize_cxcywh_boxes(boxes, eps=1e-6):
    boxes = torch.nan_to_num(boxes, nan=0.5, posinf=1.0, neginf=0.0)
    xy = boxes[..., :2].clamp(0.0, 1.0)
    wh = boxes[..., 2:].clamp(min=eps, max=1.0)
    return torch.cat([xy, wh], dim=-1)


def sanitize_logits(x, limit=30.0):
    x = torch.nan_to_num(x, nan=0.0, posinf=limit, neginf=-limit)
    return x.clamp(min=-limit, max=limit)


class SetCriterion(nn.Module):
    """Compute detector losses and the small OT embedding regularizer."""
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses, focal_alpha=0.25):
        """Initialize the criterion and its loss coefficients."""
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses

        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

        self.focal_alpha = focal_alpha
        self.mask_out_stride = 4


    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification focal loss."""
        assert 'pred_logits' in outputs
        src_logits = sanitize_logits(outputs['pred_logits'])
        _, nf, nq = src_logits.shape[:3]
        src_logits = rearrange(src_logits, 'b t q k -> b (t q) k')
        valid_indices = []
        valids = [target['valid'] for target in targets]
        for valid, (indice_i, indice_j) in zip(valids, indices):
            valid_ind = valid.nonzero().flatten()
            valid_i = valid_ind * nq + indice_i
            valid_j = valid_ind + indice_j * nf
            valid_indices.append((valid_i, valid_j))

        idx = self._get_src_permutation_idx(valid_indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, valid_indices)])

        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        if self.num_classes == 1:
            target_classes[idx] = 0
        else:
            target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_classes_onehot = target_classes_onehot[:, :, :-1]

        loss_ce = sigmoid_focal_loss(
            src_logits, target_classes_onehot, num_boxes,
            alpha=self.focal_alpha, gamma=2
        ) * src_logits.shape[1]

        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Bounding-box L1 and GIoU losses."""
        assert 'pred_boxes' in outputs
        src_boxes = outputs['pred_boxes']
        src_boxes = src_boxes.transpose(1, 2)

        idx = self._get_src_permutation_idx(indices)
        src_boxes = src_boxes[idx]
        src_boxes = sanitize_cxcywh_boxes(src_boxes.flatten(0, 1))

        target_boxes = sanitize_cxcywh_boxes(torch.cat([t['boxes'] for t in targets], dim=0))

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {'loss_bbox': loss_bbox.sum() / num_boxes}

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)
        ))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses
    def loss_video_cls(self, outputs, targets, indices, num_boxes, **kwargs):
        if 'video_logits' not in outputs:
            return {}
        if isinstance(targets, dict) and "classes" in targets:
            video_labels = targets["classes"]
        elif isinstance(targets, list):
            video_labels = torch.stack([t["classes"][0] for t in targets])
        else:
            raise RuntimeError("Targets must contain class labels")
        video_logits = sanitize_logits(outputs["video_logits"])
        return {'loss_video_cls': F.cross_entropy(video_logits, video_labels)}
    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Mask focal, Dice, and projection losses."""
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)

        src_masks = sanitize_logits(outputs["pred_masks"])
        src_masks = src_masks.transpose(1, 2)
        src_masks = src_masks[src_idx]

        target_masks, valid = nested_tensor_from_tensor_list(
            [t["masks"] for t in targets], size_divisibility=32, split=False
        ).decompose()
        target_masks = torch.nan_to_num(target_masks.to(src_masks), nan=0.0, posinf=1.0, neginf=0.0)

        start = int(self.mask_out_stride // 2)
        im_h, im_w = target_masks.shape[-2:]
        nf = target_masks.shape[1]

        target_masks = target_masks[:, :, start::self.mask_out_stride, start::self.mask_out_stride]
        valid = valid[:, start::self.mask_out_stride, start::self.mask_out_stride]
        valid = valid[:, None].expand(-1, nf, -1, -1)
        assert target_masks.size(2) * self.mask_out_stride == im_h
        assert target_masks.size(3) * self.mask_out_stride == im_w

        src_y = src_masks.max(dim=2)[0].flatten(1)
        src_x = src_masks.max(dim=3)[0].flatten(1)
        tgt_y = target_masks.max(dim=2)[0].flatten(1)
        tgt_x = target_masks.max(dim=3)[0].flatten(1)

        src_masks = src_masks.flatten(1)
        target_masks = target_masks.flatten(1)
        valid = valid.flatten(1)

        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes, valid=valid),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes, valid=valid),
            "loss_proj": (dice_loss(src_y, tgt_y, num_boxes) + dice_loss(src_x, tgt_x, num_boxes)) / 2
        }
        return losses


    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'video_cls': self.loss_video_cls
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def _loss_ot_regularizer(self, outputs: dict) -> dict:
        """Apply a small L2 regularizer to the global OT-Prompt vector."""
        losses = {}
        if 'loss_ot_reg' not in self.weight_dict:
            return losses

        hat_T = outputs.get('hat_T_global', None)
        if hat_T is None:
            return losses

        hat_T = torch.nan_to_num(hat_T.float(), nan=0.0, posinf=1.0, neginf=-1.0)
        losses['loss_ot_reg'] = hat_T.pow(2).mean()
        return losses

    def forward(self, outputs, targets):
        """Aggregate detector losses and the OT regularizer."""
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        indices = self.matcher(outputs_without_aux, targets)

        target_valid = torch.stack([t["valid"] for t in targets], dim=0).reshape(-1)
        num_boxes = target_valid.sum().item()
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float,
                                    device=next(iter(outputs.values())).device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {}
                    if loss == 'labels':
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'interm_outputs' in outputs:
            indices = self.matcher(outputs['interm_outputs_for_matching_pre'], targets)
            for loss in self.losses:
                kwargs = {}
                if loss == 'labels':
                    kwargs = {'log': False}
                l_dict = self.get_loss(loss, outputs['interm_outputs'], targets, indices, num_boxes, **kwargs)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)


        try:
            losses.update(self._loss_ot_regularizer(outputs))
        except Exception:
            pass




        return losses
