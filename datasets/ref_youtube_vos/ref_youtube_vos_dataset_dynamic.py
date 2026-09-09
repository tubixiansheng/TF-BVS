













































import os
import json
import random
from typing import List, Dict, Any, Tuple, Optional

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as VF
from torchvision.transforms import InterpolationMode

DEFAULT_SIZE = 256
IM_MEAN = [0.2663, 0.2736, 0.2876]
IM_STD  = [0.1378, 0.1416, 0.1483]


def _to_int(fid: str) -> int:
    return int(fid)


def _to_str(idx: int, z: int = 5) -> str:
    return f"{idx:0{z}d}"


def _load_mask_bin(path: str) -> np.ndarray:
    """Load a binary mask from a single-instance or nonzero-valued image."""
    m = Image.open(path).convert("L")
    arr = np.array(m)
    vals = np.unique(arr)
    if set(vals.tolist()) <= {0, 255}:
        fg = (arr == 255)
    else:
        fg = (arr > 0)
    return fg.astype(np.float32)


class BranchAugment:
    """Apply image and mask-consistent augmentations for branch training."""
    def __init__(self, size: int = DEFAULT_SIZE, hflip_prob: float = 0.5,
                 mean: List[float] = IM_MEAN, std: List[float] = IM_STD):
        self.size = size
        self.hflip_prob = hflip_prob
        self.mean = mean
        self.std = std

    def _resize_img(self, img: Image.Image) -> Image.Image:
        return VF.resize(img, self.size, interpolation=InterpolationMode.BILINEAR)

    def _resize_mask(self, mask_np: np.ndarray) -> torch.Tensor:
        mask_pil = Image.fromarray((mask_np * 255).astype(np.uint8), mode='L')
        mask_resized = VF.resize(mask_pil, self.size, interpolation=InterpolationMode.NEAREST)
        mask_t = torch.from_numpy(np.array(mask_resized)).float() / 255.0
        mask_t = (mask_t > 0.5).float()
        return mask_t

    def __call__(self, imgs: List[Image.Image], mask_a_np: np.ndarray, mask_b_np: np.ndarray):
        imgs_resized = [self._resize_img(im) for im in imgs]
        mask_a = self._resize_mask(mask_a_np)
        mask_b = self._resize_mask(mask_b_np)

        do_flip = (random.random() < self.hflip_prob)
        if do_flip:
            imgs_resized = [VF.hflip(im) for im in imgs_resized]
            mask_a = torch.flip(mask_a, dims=[1])
            mask_b = torch.flip(mask_b, dims=[1])

        imgs_t = [VF.to_tensor(im) for im in imgs_resized]
        imgs_t = [VF.normalize(t, mean=self.mean, std=self.std) for t in imgs_t]

        return imgs_t, mask_a, mask_b


def _corrupt_mask_tensor(mask_t: torch.Tensor, prob_erase: float = 0.2, max_erase_frac: float = 0.3) -> torch.Tensor:
    """Apply small random erasures to a binary mask."""
    if random.random() >= prob_erase:
        return mask_t

    m = mask_t.clone()
    H, W = m.shape
    ys, xs = torch.where(m > 0.5)
    if ys.numel() == 0:
        return m
    y0, y1 = int(ys.min().item()), int(ys.max().item()) + 1
    x0, x1 = int(xs.min().item()), int(xs.max().item()) + 1

    max_h = max(1, int((y1 - y0) * max_erase_frac))
    max_w = max(1, int((x1 - x0) * max_erase_frac))
    er_h = random.randint(1, max_h)
    er_w = random.randint(1, max_w)
    sy = random.randint(y0, max(y0, y1 - er_h))
    sx = random.randint(x0, max(x0, x1 - er_w))
    ey = min(H, sy + er_h)
    ex = min(W, sx + er_w)
    m[sy:ey, sx:ex] = 0.0

    if random.random() < 0.5:
        n_holes = random.randint(1, 3)
        for _ in range(n_holes):
            hy = random.randint(0, H - 1)
            hx = random.randint(0, W - 1)
            hh = random.randint(1, max(1, int(H * 0.02)))
            hw = random.randint(1, max(1, int(W * 0.02)))
            m[max(0, hy - hh):min(H, hy + hh), max(0, hx - hw):min(W, hx + hw)] = 0.0

    return m
def extract_grade_from_video_name(video_name):
    import re
    match = re.search(r'L([2-5])', video_name)
    if match:
        return int(match.group(1)) - 2
    else:
        raise ValueError(f"Cannot find grade label in video name: {video_name}")

class ReferYTVOSDynamicSegmentsDataset(Dataset):
    """Build dynamic clips between adjacent annotated frames."""
    def __init__(
        self,
        subset_type: str,
        dataset_path: str,
        supervise_start: bool = False,
        max_gap: Optional[int] = None,
        resize_size: int = DEFAULT_SIZE,
        hflip_prob: float = 0.5,
        corrupt_prob: float = 0.0,
    ):
        assert subset_type == "train", "DynamicSegmentsDataset is training-only"
        self.subset_type = subset_type
        self.dataset_path = dataset_path
        self.img_root = os.path.join(dataset_path, subset_type, "JPEGImages")
        self.ann_root = os.path.join(dataset_path, subset_type, "Annotations")
        self.ann_file = os.path.join(dataset_path, subset_type, "cleaned_output.json")

        self.supervise_start = supervise_start
        self.max_gap = max_gap
        self.aug = BranchAugment(size=resize_size, hflip_prob=hflip_prob)
        self.corrupt_prob = float(corrupt_prob)

        with open(self.ann_file, "r") as f:
            videos = json.load(f)["videos"]

        self.obj2exp = {}
        for vid, vinfo in videos.items():
            obj_exps = {}
            if "expressions" in vinfo:
                for exp_id, exp_info in vinfo["expressions"].items():
                    obj_id = str(exp_info["obj_id"])
                    exp_text = exp_info["exp"]
                    obj_exps[obj_id] = exp_text
            self.obj2exp[vid] = obj_exps

        self.segments: List[Tuple[str, str, str, str]] = []
        total_pairs, skipped_long, skipped_missing = 0, 0, 0

        for vid, vinfo in videos.items():
            frames_set = set(vinfo.get("frames", []))
            for obj_id, oinfo in vinfo.get("objects", {}).items():
                valid = [f for f in oinfo.get("valid_frames", []) if f in frames_set]
                valid = sorted(valid, key=_to_int)
                for i in range(len(valid) - 1):
                    total_pairs += 1
                    a, b = valid[i], valid[i + 1]
                    gap = _to_int(b) - _to_int(a)
                    if (self.max_gap is not None) and (gap > self.max_gap):
                        skipped_long += 1
                        continue
                    mask_a_path = os.path.join(self.ann_root, vid, f"{a}.png")
                    mask_b_path = os.path.join(self.ann_root, vid, f"{b}.png")
                    if (not os.path.exists(mask_a_path)) or (not os.path.exists(mask_b_path)):
                        skipped_missing += 1
                        continue
                    self.segments.append((vid, obj_id, a, b))

        print(f"[DynamicSegments] total_pairs={total_pairs} kept={len(self.segments)} "
              f"skipped_long={skipped_long} skipped_missing={skipped_missing} "
              f"(max_gap={self.max_gap}) corrupt_prob={self.corrupt_prob}")

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], Dict[str, Any], str]:
        vid, obj_id, a_str, b_str = self.segments[idx]
        a_i, b_i = _to_int(a_str), _to_int(b_str)
        frame_ids = [_to_str(t) for t in range(a_i, b_i + 1)]
        classes = extract_grade_from_video_name(vid)
        imgs = [Image.open(os.path.join(self.img_root, vid, f"{fid}.jpg")).convert("RGB") for fid in frame_ids]
        w, h = imgs[0].size

        mask_a_np = _load_mask_bin(os.path.join(self.ann_root, vid, f"{a_str}.png"))
        mask_b_np = _load_mask_bin(os.path.join(self.ann_root, vid, f"{b_str}.png"))

        imgs_aug, mask_a_t, mask_b_t = self.aug(imgs, mask_a_np, mask_b_np)

        if self.corrupt_prob > 0.0:
            mask_a_t = _corrupt_mask_tensor(mask_a_t, prob_erase=self.corrupt_prob)
            mask_b_t = _corrupt_mask_tensor(mask_b_t, prob_erase=self.corrupt_prob)

        if self.supervise_start:
            mask_sup = mask_a_t
            sup_fid = a_str
        else:
            mask_sup = mask_b_t
            sup_fid = b_str

        Ht, Wt = imgs_aug[0].shape[-2], imgs_aug[0].shape[-1]

        caption = None
        obj_exps = self.obj2exp.get(vid, {})
        if obj_exps:
            caption = obj_exps.get(str(obj_id), None)

        target = {
            "video": vid,
            "obj_id": int(obj_id) if str(obj_id).isdigit() else obj_id,
            "frame_ids": frame_ids,
            "start_fid": a_str,
            "end_fid": b_str,
            "supervise_fid": sup_fid,
            "mask_start": mask_a_t,
            "mask_end": mask_b_t,
            "mask_sup": mask_sup,
            "caption": caption,
            "orig_size": torch.as_tensor([h, w]),
            "size": torch.as_tensor([Ht, Wt]),
            "classes": classes,
        }
        return imgs_aug, target, vid


def dynamic_collate_for_branch(batch: List[Tuple[List[torch.Tensor], Dict[str, Any], str]]) -> Dict[str, Any]:
    """Pad variable-length dynamic clips into a batch."""
    B = len(batch)
    T_list = [len(x[0]) for x in batch]
    T_max = max(T_list)
    C, H, W = batch[0][0][0].shape

    imgs = torch.zeros(B, T_max, C, H, W, dtype=batch[0][0][0].dtype)
    valid_mask = torch.zeros(B, T_max, dtype=torch.bool)
    for i, (imgs_list, _, _) in enumerate(batch):
        T = len(imgs_list)
        imgs[i, :T] = torch.stack(imgs_list, dim=0)
        valid_mask[i, :T] = True

    mem_img = imgs[:, 0]
    frames = imgs[:, 1:]
    valid_time_mask = valid_mask[:, 1:]

    mask_start = torch.stack([b[1]["mask_start"] for b in batch], dim=0)[:, None].float()
    mask_end   = torch.stack([b[1]["mask_end"]   for b in batch], dim=0)[:, None].float()
    mask_sup   = torch.stack([b[1]["mask_sup"]   for b in batch], dim=0)[:, None].float()
    text_queries = [b[1]["caption"] for b in batch]
    classes = torch.tensor([b[1]["classes"] for b in batch], dtype=torch.long)
    out = {
        "video_id":        [b[1]["video"] for b in batch],
        "frame_ids":       [b[1]["frame_ids"] for b in batch],
        "start_fid":       [b[1]["start_fid"] for b in batch],
        "end_fid":         [b[1]["end_fid"] for b in batch],
        "supervise_fid":   [b[1]["supervise_fid"] for b in batch],
        "mem_img":         mem_img,
        "frames":          frames,
        "valid_time_mask": valid_time_mask,
        "mask_start":      mask_start,
        "mask_end":        mask_end,
        "mask_sup":        mask_sup,
        "text_queries":    text_queries,
        "H": H, "W": W,
        "classes": classes,
    }
    return out
