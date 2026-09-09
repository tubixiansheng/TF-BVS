import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from medpy.metric.binary import hd95, assd

def dice_score(gt, pred):
    """Dice coefficient."""
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    intersection = np.logical_and(gt, pred).sum()
    total = gt.sum() + pred.sum()
    if total == 0:
        return 1.0
    return 2. * intersection / total

def db_eval_iou(annotation, segmentation, void_pixels=None):
    assert annotation.shape == segmentation.shape
    annotation = annotation.astype(bool)
    segmentation = segmentation.astype(bool)

    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(segmentation, dtype=bool)

    inters = np.sum((segmentation & annotation) & ~void_pixels)
    union = np.sum((segmentation | annotation) & ~void_pixels)

    j = inters / union if union != 0 else 1
    return j

def db_eval_boundary(annotation, segmentation, void_pixels=None, bound_th=0.008):
    assert annotation.shape == segmentation.shape
    if void_pixels is not None:
        assert annotation.shape == void_pixels.shape
    if annotation.ndim == 2:
        return f_measure(segmentation, annotation, void_pixels, bound_th)
    else:
        raise ValueError("Only 2D masks are supported.")

def f_measure(foreground_mask, gt_mask, void_pixels=None, bound_th=0.008):
    assert np.atleast_3d(foreground_mask).shape[2] == 1
    if void_pixels is not None:
        void_pixels = void_pixels.astype(bool)
    else:
        void_pixels = np.zeros_like(foreground_mask, dtype=bool)

    bound_pix = bound_th if bound_th >= 1 else \
        np.ceil(bound_th * np.linalg.norm(foreground_mask.shape))

    fg_boundary = _seg2bmap(foreground_mask * (~void_pixels))
    gt_boundary = _seg2bmap(gt_mask * (~void_pixels))

    from skimage.morphology import disk
    fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(int(bound_pix)).astype(np.uint8))
    gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(int(bound_pix)).astype(np.uint8))

    gt_match = gt_boundary * fg_dil
    fg_match = fg_boundary * gt_dil

    n_fg = np.sum(fg_boundary)
    n_gt = np.sum(gt_boundary)

    if n_fg == 0 and n_gt > 0:
        precision = 1
        recall = 0
    elif n_fg > 0 and n_gt == 0:
        precision = 0
        recall = 1
    elif n_fg == 0 and n_gt == 0:
        precision = 1
        recall = 1
    else:
        precision = np.sum(fg_match) / float(n_fg)
        recall = np.sum(gt_match) / float(n_gt)

    F = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
    return F

def _seg2bmap(seg, width=None, height=None):
    seg = seg.astype(bool)
    seg[seg > 0] = 1
    assert np.atleast_3d(seg).shape[2] == 1

    width = seg.shape[1] if width is None else width
    height = seg.shape[0] if height is None else height
    h, w = seg.shape[:2]

    assert not (width > w or height > h or abs((float(width) / height) - (float(w) / h)) > 0.01)

    e = np.zeros_like(seg)
    s = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:, :-1] = seg[:, 1:]
    s[:-1, :] = seg[1:, :]
    se[:-1, :-1] = seg[1:, 1:]

    b = seg ^ e | seg ^ s | seg ^ se
    b[-1, :] = seg[-1, :] ^ e[-1, :]
    b[:, -1] = seg[:, -1] ^ s[:, -1]
    b[-1, -1] = 0

    return b

def is_non_empty_mask(mask):
    return np.any(mask > 0)

def evaluate_all(gt_root, pred_root, csv_out_path="results.csv"):
    records = []

    video_list = sorted(os.listdir(gt_root))

    for video in tqdm(video_list, desc="Evaluating video segmentation"):
        gt_dir = os.path.join(gt_root, video)
        pred_dir = os.path.join(pred_root, video)

        if not os.path.isdir(gt_dir) or not os.path.isdir(pred_dir):
            continue

        frame_names = sorted(os.listdir(gt_dir))

        for fname in frame_names:
            gt_path = os.path.join(gt_dir, fname)
            pred_path = os.path.join(pred_dir,'exp0', fname)

            if not os.path.exists(pred_path):
                continue

            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

            if gt is None or pred is None or not is_non_empty_mask(gt):
                continue

            gt_bin = (gt > 0).astype(np.uint8)
            pred_bin = (pred > 0).astype(np.uint8)

            iou = db_eval_iou(gt_bin, pred_bin)
            fscore = db_eval_boundary(gt_bin, pred_bin)
            dice = dice_score(gt_bin, pred_bin)
            try:
                haus95 = hd95(pred_bin, gt_bin)
            except Exception:
                haus95 = np.nan
            try:
                asd = assd(pred_bin, gt_bin)
            except Exception:
                asd = np.nan

            records.append({
                "video": video,
                "frame": fname,
                "IoU": iou,
                "F-measure": fscore,
                "Dice": dice,
                "HD95": haus95,
                "ASD": asd,
            })

    df = pd.DataFrame(records)
    df.to_csv(csv_out_path, index=False)

    print(f"\nMetrics saved to: {csv_out_path}")
    print(f"Evaluated frames: {len(df)}")
    print(f"Mean IoU: {df['IoU'].mean():.4f}")
    print(f"Mean F-measure: {df['F-measure'].mean():.4f}")
    print(f"Mean Dice: {df['Dice'].mean():.4f}")
    print(f"Mean HD95: {df['HD95'].mean():.4f}")
    print(f"Mean ASD: {df['ASD'].mean():.4f}")

if __name__ == "__main__":
    evaluate_all(
        gt_root="data/processed_breast_ultrasound_5frames/test/Annotations",
        pred_root="output/ref_youtube_vos/branch_fusion/Annotations",
        csv_out_path="output/ref_youtube_vos/branch_fusion/results_test.csv"
    )
