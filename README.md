<div align="center">

# TF-BVS

### Text-Free Breast Ultrasound Video Segmentation

<p>
  <a href="https://2026.acmmm.org/"><img src="https://img.shields.io/badge/ACM%20MM-2026%20Accepted-8A2BE2?style=for-the-badge" alt="ACM MM 2026 Accepted"></a>
  <img src="https://img.shields.io/badge/Task-Breast%20Ultrasound%20VOS-0ea5e9?style=for-the-badge" alt="Breast Ultrasound VOS">
  <img src="https://img.shields.io/badge/Mode-Text--Free%20Inference-22c55e?style=for-the-badge" alt="Text-Free Inference">
</p>

<p>
  <b>No caption at inference time? No problem.</b><br>
  TF-BVS turns visual-temporal evidence into a robust segmentation prompt.
</p>

<p>
  <a href="#quick-start">Quick Start</a> ·
  <a href="#citation">Citation</a>
</p>

</div>

> 🎉 Our paper **“A Large-Scale Multi-Modal Benchmark and Robust Text-Free Inference for Breast Ultrasound Video Segmentation”** has been accepted by **ACM Multimedia 2026**.

## Model overview

<p align="center">
  <img src="assets/tf-bvs-model-architecture.png" alt="TF-BVS model architecture" width="100%">
</p>

<p align="center"><em>TF-BVS: cross-frame backbone, frame-wise temporal branch, and LV2TOT prompt learning.</em></p>

<p align="center">
  <a href="assets/tf-bvs-model-architecture.pdf">Download the original PDF figure</a>
</p>

## Quick start

### 1. Install

```bash
conda create -n tfbvs python=3.10
conda activate tfbvs
pip install -r requirements.txt
```

Build the Deformable Attention extension:

```bash
cd models/GroundingDINO/ops
python setup.py build install
cd ../../
```

Prepare these resources locally. They are intentionally not included in the repository:

- processed MBTUVS data with `train`, `valid`, and `test` splits;
- a local `bert-base-uncased` snapshot;
- the Swin-B COCO checkpoint;
- the GroundingDINO Swin-B checkpoint.

```bash
cd /path/to/TF-BVS
export PYTHONPATH=.
export DATA_ROOT=/path/to/processed_breast_ultrasound_5frames
export TEXT_ENCODER_PATH=/path/to/bert-base-uncased
export COCO_WEIGHTS=/path/to/coco_swinb.pth
export DINO_WEIGHTS=/path/to/groundingdino_swinb_cogcoor.pth
export TRANSFORMERS_NO_TF=1
export USE_TF=0
```

Set `GroundingDINO.pretrained_path` and `dataset_path` in `configs/ytvos_swinb.yaml` to the corresponding local resources.

### 2. Train with validation-based checkpoint selection

The validation wrapper trains one epoch at a time, evaluates the `valid` split after each epoch, and keeps the checkpoint with the best combined score. The test split is never used for model selection.

```bash
python scripts/rebuttal/train_with_validation.py \
  --total-epochs 60 \
  --validation-dataset-path "$DATA_ROOT" \
  --validation-dataset-root "$DATA_ROOT" \
  --validation-split valid \
  --validation-num-gpus 4 \
  --frame-step 5 \
  --score-weights 0.5 0.3 0.2 \
  -- python main.py \
    -c configs/ytvos_swinb.yaml -rm train -ng 4 -bs 2 \
    --epochs 60 --version tfbvs_online_valid \
    --lr_drop 30 --lr_scale 0.25 --enable_amp false \
    --pretrained_weights "$COCO_WEIGHTS" \
    --use_ot_prompt true --textfree_infer true --textless_prob 0.5 \
    --w_ot 5.0 --w_contrast 0.0 --w_pair 4.0 --w_mmd 2.0 --w_tv 1.0 \
    --w_grade_contrast 0.0 --ot_eps 0.5 --ot_iters 50 \
    --mmd_sigma 1.0 --mmd_ref_num 512 --mmd_bank_size 8192
```

Head-init supervision is enabled by default in `configs/ytvos_swinb.yaml` with `head_sup_weight=1.0` and `head_sup_epochs=60`.

The selected checkpoint is written to:

```text
output/ref_youtube_vos/tfbvs_online_valid/checkpoints/best_val.pth.tar
```

### 3. Run inference and evaluate

```bash
python scripts/rebuttal/run_full_inference_pipeline.py \
  -c configs/ytvos_swinb.yaml -ng 4 \
  -ckpt output/ref_youtube_vos/tfbvs_online_valid/checkpoints/best_val.pth.tar \
  --dataset-path "$DATA_ROOT" --dataset-root "$DATA_ROOT" --split test \
  --backbone-version tfbvs_test_backbone \
  --fusion-version tfbvs_test_fusion \
  --frame-step 5 \
  --metrics-name tfbvs_test \
  --output-dir outputs/metrics
```

The unified summary is written to:

```text
outputs/metrics/tfbvs_test_meter_summary.csv
```

To evaluate a corrected test set, replace both dataset arguments and `--split` with the corrected dataset root and split name.

## Project map

```text
configs/ytvos_swinb.yaml                    # reproducible Swin-B configuration
main.py                                     # training entry point
trainer.py                                  # optimization and temporal training
scripts/rebuttal/train_with_validation.py   # online validation and selection
scripts/rebuttal/run_full_inference_pipeline.py # inference + fusion + metrics
scripts/rebuttal/evaluate_meter_mbtuvs.py   # unified evaluator
```

## Citation

## Notes

- Please follow the licenses and terms of the upstream GroundingDINO, COCO, and dataset resources.
- If TF-BVS helps your research, consider ⭐ starring the repository and citing the paper.

<div align="center">

**From pixels → prompts → temporally consistent masks.** 🚀

</div>
