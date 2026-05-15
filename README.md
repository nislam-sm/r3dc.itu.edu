<div align="center">

# R³DC: Reliability-Guided Reveal-to-Revise Depth Completion<br/>for Cross-Domain Sparse Perception

[![CVPR 2026 Workshop](https://img.shields.io/badge/CVPR%202026-3D%20Geometry%20Generation%20Workshop-1f6feb.svg)](https://openreview.net/forum?id=odj32HFuaj)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-d63232.svg)](https://openreview.net/forum?id=odj32HFuaj)
[![License: MIT](https://img.shields.io/badge/License-MIT-2ea44f.svg)](LICENSE)
[![Python ≥3.9](https://img.shields.io/badge/Python-%E2%89%A53.9-3776ab.svg)](https://www.python.org/)
[![PyTorch ≥2.0](https://img.shields.io/badge/PyTorch-%E2%89%A52.0-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Official PyTorch implementation** of the CVPR 2026 Workshop paper by
[Noor Islam S. Mohammad](https://openreview.net/profile?id=~Noor_Islam_S_Mohammad1) and [Uluğ Beyazıt](https://openreview.net/profile?id=~Ulug_Beyazit1) (Istanbul Technical University).

R³DC is a **Reveal-to-Revise** depth-completion framework that jointly predicts dense metric depth, per-pixel **reliability**, and aleatoric **uncertainty** across four heterogeneous benchmarks — KITTI, VisDrone, Drone-Videos, and NYU Depth V2 — with one fixed loss recipe and **6–67× fewer parameters** than depth-completion baselines.

[📄 Paper](https://openreview.net/forum?id=odj32HFuaj) ·
[🚀 Quick Start](#-quick-start) ·
[🧠 Method](#-method-overview) ·
[🔬 RADI](#-radi--the-reliability-aware-depth-index) ·
[📊 Results](#-results) ·
[🧪 Repro](#-reproducibility) ·
[📝 Cite](#-citation)

</div>

---

## Contents

1. [Highlights](#-highlights)
2. [Method overview](#-method-overview)
3. [Repository layout](#-repository-layout)
4. [Installation](#-installation)
5. [Quick start](#-quick-start)
6. [Datasets](#-datasets)
7. [Training](#-training)
8. [Evaluation](#-evaluation)
9. [RADI — the Reliability-Aware Depth Index](#-radi--the-reliability-aware-depth-index)
10. [Model zoo](#-model-zoo)
11. [Results](#-results)
12. [Reproducibility](#-reproducibility)
13. [FAQ](#-faq)
14. [Limitations](#-limitations)
15. [Citation](#-citation)
16. [License & acknowledgements](#-license--acknowledgements)

---

## ✨ Highlights

- **One paradigm, four domains.** A single *Reveal-to-Revise* design that *reveals* a coarse depth from a global encoder–decoder, then *revises* it through reliability-gated spatial propagation. Two encoder instantiations cover four domains (KITTI / VisDrone / Drone-Videos / NYU).
- **Reliability as a first-class output.** Each prediction comes with a learned per-pixel reliability map that gates CSPN++ propagation, exposes failure regions, and is independently scored by **RADI** (see below).
- **Parameter-efficient.** **1.95 M** params for outdoor KITTI, **1.47 M** for Drone-Videos, **11.22 M** for the v3 VisDrone variant, and **16,642 trainable params** for the NYU adapter on top of a frozen Depth-Anything-V2 ViT-S backbone — **6–67× fewer** than depth-completion baselines.
- **A new reliability metric: RADI.** Composite metric combining **REC** (rank correlation), **RBS** (refinement benefit by reliability stratum), and **CAL** (ECE-style calibration). On NYU, R³DC achieves **ρ ≈ +0.43 (p < 0.001) across all regions**, **RBS > 62 %**, and **ECE = 0.031**.
- **Honest evaluation.** We separately report results under both relative-metres and the official KITTI mm protocol (786.4 mm RMSE — competitive but not SOTA), and we explicitly mark RADI on the aerial benchmarks as *provisional* because the synthetic ground truth is smooth at object boundaries.
- **Open & reproducible.** Configs, scripts, deterministic seeds, EMA weights, AMP, DDP, and a self-contained synthetic-GT generator are all in this repo.

---

## 🧠 Method overview

```text
                              ┌──────────────────────────────────────────┐
                              │           Reveal-to-Revise               │
                              └──────────────────────────────────────────┘

  RGB ──► RGB-stream encoder ─┐                       ┌─► Aux head (1/2) ─► aux_depth_half
                              ├── Cross-Modal ────────┤
  Sparse depth                │   Attention (CMA)     ├─► Aux head (1/4) ─► aux_depth_quarter
   + sparse mask              │   @ s2 / s4 / s8      │
            └─► Depth-stream encoder ─────────────────┤
                  (DCNv2 in stride-2 layers)          │
                                                      ▼
                                          Transformer bottleneck (1/16)
                                                      │
                                          FPN decoder (s8 → s4 → s2 → full)
                                                      │
                                  ┌───────────────────┴────────────────────┐
                                  ▼                                        ▼
                          Depth head (sigmoid)                Reliability head (sigmoid)
                                  │                                        │
                                  └──────► coarse_depth ──┐                │
                                                          ▼                ▼
                              Sparse anchors ─► Reliability-gated CSPN++ (T = 6)
                                                          │
                                                          ▼
                                       Refined dense metric depth + reliability
                                                + aleatoric uncertainty (softplus)
```

**Outdoor variant (`R³DC`)** — dual-stream encoder shared across KITTI / VisDrone / Drone-Videos, 1.47–11.22 M params depending on base-channel width `B ∈ {32, 64}`.

**Indoor variant (`R³DC+ICH`)** — backbone is a *frozen* Depth-Anything-V2 ViT-S; only the lightweight Indoor Calibration Head (3-layer MLP, **16,642 trainable params**), reliability head, and refiner are trained on NYU.

**Training objective.** A seven-term composite loss with **fixed weights across all four datasets** (Eq. 10 in the paper):

```
L = 1.00·L_SILog + 0.60·L_FocalBerHu + 0.20·L_SSIM + 0.15·L_anchor
  + 0.10·L_VNL  + 0.05·L_DNC        + 0.05·L_grad  + 0.05·L_unc
  + 0.10·L_aux
```

---

## 🗂 Repository layout

```text
R3DC/
├── r3dc/
│   ├── models/        # encoder, CMA, transformer bottleneck, FPN decoder,
│   │                  # CSPN refiner, heads, ICH adapter, full R3DC model
│   ├── losses/        # seven composite-loss terms + aggregator
│   ├── metrics/       # standard depth metrics + RADI (REC/RBS/CAL)
│   ├── datasets/      # KITTI, VisDrone, Drone-Videos, NYU + augmentations
│   ├── utils/         # log-norm, EMA, seeding, viz, logging, DDP helpers
│   ├── configs/       # YAML configs: kitti / visdrone / drone_videos / nyu
│   └── cli.py         # console entry points (r3dc-train / -eval / -infer)
├── scripts/
│   ├── train.py             # main training entry (single-GPU + DDP)
│   ├── eval.py              # standard metrics + RADI evaluator
│   ├── infer.py             # single-image inference + reliability overlay
│   ├── make_aerial_gt.py    # pre-generate synthetic depth for VisDrone/Drone-Videos
│   ├── benchmark_speed.py   # FPS / latency benchmark
│   └── export_onnx.py
├── tests/             # unit tests (lognorm, losses, RADI, forward pass)
├── docs/              # INSTALL, DATASETS, TRAINING, METRICS, RADI, MODEL_ZOO
├── examples/          # quick-start notebooks
├── assets/            # figures / diagrams
├── pyproject.toml
├── requirements.txt
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
└── README.md
```

---

## 🛠 Installation

R³DC targets **Python ≥ 3.9** and **PyTorch ≥ 2.0** with CUDA 11.8+.

```bash
git clone https://github.com/<user>/r3dc.git
cd r3dc

# 1. (recommended) create a fresh environment
conda create -n r3dc python=3.10 -y
conda activate r3dc

# 2. install PyTorch matching your CUDA version, e.g.
pip install torch==2.3.1+cu118 torchvision==0.18.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 3. install R³DC + runtime dependencies
pip install -e .

# 4. (optional) developer extras
pip install -r requirements-dev.txt
pre-commit install
```

After installation the following console scripts are available:

```bash
r3dc-train  --config r3dc/configs/kitti.yaml
r3dc-eval   --config r3dc/configs/kitti.yaml --checkpoint runs/kitti/best.pt
r3dc-infer  --config r3dc/configs/kitti.yaml --checkpoint runs/kitti/best.pt \
            --rgb path/to/image.png --sparse path/to/sparse.png
```

See [`docs/INSTALL.md`](docs/INSTALL.md) for CPU-only / Apple Silicon notes and known caveats.

---

## 🚀 Quick start

A **30-second demo** using random sparse anchors on any RGB image:

```bash
r3dc-infer \
    --config r3dc/configs/kitti.yaml \
    --checkpoint checkpoints/r3dc_kitti.pt \
    --rgb demo/scene.png \
    --output-dir demo_out
```

This writes:

- `depth.png` / `depth.npy` — predicted dense metric depth (magma colormap)
- `reliability.png` / `reliability.npy` — per-pixel reliability map (RdYlGn)
- `uncertainty.npy` — aleatoric uncertainty (softplus)

**Train on KITTI** (single GPU):

```bash
r3dc-train --config r3dc/configs/kitti.yaml --output-dir runs/kitti
```

**Train on NYU with 2× T4 DDP** (R³DC + ICH adapter on frozen Depth-Anything-V2):

```bash
torchrun --nproc_per_node=2 -m scripts.train \
    --config r3dc/configs/nyu.yaml --ddp --output-dir runs/nyu
```

**Evaluate with RADI** (all four region splits):

```bash
r3dc-eval \
    --config r3dc/configs/nyu.yaml \
    --checkpoint runs/nyu/best.pt \
    --metrics standard,radi \
    --regions all,edge,textureless,far \
    --save-qualitative runs/nyu/qual \
    --output runs/nyu/metrics.json
```

---

## 📦 Datasets

R³DC supports four benchmarks. Detailed download + layout instructions live in [`docs/DATASETS.md`](docs/DATASETS.md); a short summary:

| Dataset            | Resolution   | Depth range   | Sparse density | GT source              | Loader                          |
|--------------------|--------------|---------------|----------------|------------------------|---------------------------------|
| **KITTI DC**       | 352 × 1216   | 0–80 m        | ~5 % (LiDAR)   | LiDAR (real)           | `KITTIDepthCompletion`          |
| **VisDrone**       | 384 × 640    | 1–80 m        | ~2.5 %         | **Synthetic** (App. J) | `VisDroneDataset`               |
| **Drone-Videos**   | 384 × 640    | 0–50 m        | ~2.5 %         | **Synthetic** (App. J) | `DroneVideosDataset`            |
| **NYU Depth V2**   | 518 × 518    | 0.001–10 m    | ~0.1 %         | Structured light       | `NYUDepthV2`                    |

> **Note on aerial GT.** VisDrone and Drone-Videos do not ship real aerial LiDAR. We use the physics-motivated synthetic prior from Appendix J of the paper, implemented in [`r3dc/datasets/synthetic.py`](r3dc/datasets/synthetic.py). Pre-generate once for speed:
>
> ```bash
> python -m scripts.make_aerial_gt \
>     --rgb-dir /datasets/visdrone/VisDrone2019-DET-train/images \
>     --cache-dir /datasets/visdrone/_r3dc_cache \
>     --d-min 1 --d-max 80
> ```

---

## 🏋 Training

All four recipes share the same composite-loss weights; only the encoder width, depth range, schedule, and resolution differ. Each config is a single YAML file under `r3dc/configs/`.

| Config              | Size              | Params  | Optim       | LR     | Epochs | Schedule                       | Batch |
|---------------------|-------------------|---------|-------------|--------|--------|--------------------------------|-------|
| `kitti.yaml`        | 352 × 1216        | 1.95 M  | AdamW       | 1e-4   | 8      | CosineWarmRestarts (T₀=10)     | 4     |
| `visdrone.yaml`     | 384 × 640         | 11.22 M | AdamW       | 1e-4   | 20     | CosWR + 5-epoch warmup         | 4     |
| `drone_videos.yaml` | 384 × 640         | 1.47 M  | AdamW       | 1e-4   | 10     | Cosine                         | 4     |
| `nyu.yaml`          | 518 × 518         | 16,642* | AdamW       | 5e-6   | 10     | Cosine + 0.5-epoch warmup      | 11**  |

\* Trainable parameters on top of a frozen DA-V2 ViT-S backbone.
\** Reported on 2× Tesla T4 with DDP.

Common knobs in every config:

- `optim.amp: true` — FP16 mixed precision via `torch.cuda.amp`
- `training.ema_decay: 0.9999` — exponential moving average of weights for evaluation
- `training.grad_clip: 1.0`
- Augmentations: hflip (p=0.5), color jitter, gamma ~ U(0.8, 1.2), sparse-anchor dropout (p=0.3), CutMix (p=0.3, λ ~ U(0.3, 0.7))

Resume a run from its last checkpoint:

```bash
r3dc-train --config r3dc/configs/kitti.yaml --resume runs/kitti/last.pt
```

Optional logging:

```bash
r3dc-train --config r3dc/configs/kitti.yaml --log-tb     # TensorBoard
r3dc-train --config r3dc/configs/kitti.yaml --log-wandb  # Weights & Biases
```

---

## 📏 Evaluation

```bash
r3dc-eval \
    --config     r3dc/configs/kitti.yaml \
    --checkpoint runs/kitti/best.pt \
    --metrics    standard,radi \
    --regions    all,edge,textureless,far \
    --output     runs/kitti/metrics.json
```

Reported standard metrics: **RMSE, MAE, AbsRel, SILog, δ₁ / δ₂ / δ₃** (with thresholds 1.25, 1.25², 1.25³). RADI is computed per region — see next section.

---

## 🔬 RADI — the Reliability-Aware Depth Index

Most depth-completion metrics ignore the question *"does the model know when it is wrong?"* — yet that is precisely what makes a network safe to deploy on a robot or vehicle. **RADI** scores three properties of the reliability output and reports them jointly:

| Component | Question it answers                                              | Definition                                                                          |
|-----------|------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| **REC**   | Does reliability *rank* with correctness?                        | Spearman rank correlation between reliability *r* and `−|d_pred − d_gt|`.           |
| **RBS**   | Does refinement *benefit more* where the model is more reliable? | % RMSE improvement (coarse → refined) within each reliability decile.               |
| **CAL**   | Is the *magnitude* of reliability calibrated?                    | ECE-style binning with tolerance τ = 0.10 on relative error.                        |

Each component is computed independently for four regions: `all`, `edge`, `textureless`, `far`.

```python
from r3dc.metrics import RADI, build_region_masks, RegionConfig

radi = RADI(num_bins=15, tolerance=0.10)
masks = build_region_masks(image, gt_depth, valid_mask, d_max=10.0, cfg=RegionConfig())
result = radi(
    reliability=outputs["reliability"],
    pred_depth=pred_metric,
    coarse_depth=coarse_metric,
    refined_depth=pred_metric,
    gt_depth=gt_depth,
    valid_mask=valid_mask,
    region_masks=masks,
)
print(result.to_dict())
# {'radi/rec/all': 0.43, 'radi/rec_p/all': 1.2e-7, 'radi/rbs/all': 0.62, 'radi/cal': 0.031, …}
```

> **⚠️ Provisional on synthetic GT.** RADI on VisDrone and Drone-Videos must be interpreted with care: because the synthetic prior is smooth at object boundaries, the residual `ε = |d_real − d_synth|` dominates at low-reliability pixels and *can* yield negative REC even when the model behaves correctly. We therefore make strong RADI claims **only on KITTI and NYU**, and tag aerial RADI numbers as provisional throughout this repo.

See [`docs/RADI.md`](docs/RADI.md) for derivations, sensitivity analyses, and recommended reporting practice.

---

## 🪄 Model zoo

Pre-trained checkpoints will be released alongside the camera-ready paper. Once published, place them under `checkpoints/`:

| Variant         | Trained on     | Params  | RMSE ↓     | δ₁ ↑      | RADI (NYU) | Download |
|-----------------|---------------|---------|------------|-----------|------------|----------|
| `R³DC`          | KITTI         | 1.95 M  | 0.240 m    | 0.947     | —          | _TBA_    |
| `R³DC` (light)  | Drone-Videos  | 1.47 M  | 0.67 m     | —         | —          | _TBA_    |
| `R³DC+ v3`      | VisDrone      | 11.22 M | 2.33 m     | 0.928     | —          | _TBA_    |
| `R³DC+ICH`      | NYU DV2       | 16,642* | 0.353 m    | 0.927     | ρ=0.43 / RBS=0.62 / ECE=0.031 | _TBA_ |

\* Trainable params only; backbone frozen. Full checksums and file sizes will appear in [`docs/MODEL_ZOO.md`](docs/MODEL_ZOO.md) when released.

---

## 📊 Results

### KITTI Depth Completion (val, relative-metres protocol)

| Method            | Params | RMSE ↓     | MAE ↓      | δ₁ ↑      |
|-------------------|--------|------------|------------|-----------|
| Baselines (see paper Table 4) | 10–60 M+ | 0.21–0.27 m | 0.07–0.10 m | 0.93–0.96 |
| **R³DC (ours)**   | **1.95 M** | **0.240 m** | 0.087 m | **0.947** |

> On the **official KITTI mm protocol**, R³DC achieves **786.4 mm RMSE**, which is competitive but *not* state-of-the-art. R³DC's value proposition is parameter efficiency and reliability-awareness, not raw mm-RMSE leadership.

### VisDrone (val, synthetic GT)

| Method            | Params  | RMSE ↓ | δ₁ ↑   |
|-------------------|---------|--------|--------|
| **R³DC+ v3**      | 11.22 M | **2.33 m** | **0.928** |

### Drone-Videos (val, synthetic GT)

| Method            | Params  | Best RMSE ↓ (epoch) |
|-------------------|---------|---------------------|
| **R³DC**          | 1.47 M  | **0.67 m** (epoch 4) |

### NYU Depth V2 (val)

| Method            | Trainable params | RMSE ↓ | AbsRel ↓ | δ₁ ↑   | REC (all) | RBS (all) | CAL (ECE) |
|-------------------|------------------|--------|----------|--------|-----------|-----------|-----------|
| **R³DC + ICH**    | 16,642           | **0.353 m** | **0.090** | **0.927** | **+0.43** (p<0.001) | **>0.62** | **0.031** |

Full ablations (each loss term, CMA, DCNv2, ICH width, augmentation) are reported in the paper's Appendix and re-generated by the configs in `r3dc/configs/`.

---

## 🧪 Reproducibility

- All RNGs are seeded via `r3dc.utils.seed_everything(seed)`; default seed = 42.
- Set `torch.backends.cudnn.deterministic=True` and `benchmark=False` for fully deterministic runs (mildly slower).
- EMA weights with decay 0.9999 are used at evaluation time — this is essential for matching the reported numbers.
- AMP/FP16 introduces ≤0.01 RMSE drift across runs; turn it off in the YAML if you need bit-exact repeats.
- Hardware used in the paper:
  - KITTI / VisDrone / Drone-Videos — single NVIDIA A100 (40 GB) or RTX 3090
  - NYU — 2× Tesla T4 (16 GB) via DDP

If you cannot reproduce a reported number within ±2 %, please [open an issue](https://github.com/<user>/r3dc/issues) with your environment dump (`python -m torch.utils.collect_env`).

---

## ❓ FAQ

**Q: Does R³DC need a pre-trained backbone?**
A: Only the *indoor* `R³DC+ICH` variant uses one (frozen Depth-Anything-V2 ViT-S). The outdoor `R³DC` is trained from scratch.

**Q: Can I plug in my own dataset?**
A: Yes. Subclass `torch.utils.data.Dataset` and return a sample dict with keys `image`, `sparse_depth`, `sparse_mask`, `depth`, `valid_mask`. See [`examples/03_custom_dataset.ipynb`](examples/03_custom_dataset.ipynb) for a worked example.

**Q: Why is the aerial ground truth synthetic?**
A: Public VisDrone and Drone-Videos datasets ship only RGB / detection annotations — there is no large-scale public aerial LiDAR at these resolutions. We use the physics-motivated prior from Appendix J and clearly mark all aerial RADI numbers as provisional.

**Q: How is RADI different from regular calibration metrics?**
A: RADI is composite: it tests *rank correlation* (REC), *refinement benefit* (RBS), **and** *bin-level magnitude calibration* (CAL). Standard ECE captures only the third.

**Q: Does R³DC support ONNX export?**
A: Yes, see [`scripts/export_onnx.py`](scripts/export_onnx.py). DCNv2 ops fall back to plain Conv2d if the deployment target does not support deformable convolutions.

---

## ⚠️ Limitations

We are deliberately explicit about what R³DC is **not**:

- **Not** state-of-the-art under the official KITTI mm protocol (786.4 mm RMSE) — see paper Sec. 5.3.
- The aerial RADI numbers are **provisional**: synthetic ground truth is smooth at object boundaries, so the residual `ε = |d_real − d_synth|` can dominate at low-reliability pixels.
- R³DC does **not** model temporal consistency. For video depth-completion, see the discussion in the paper's "Future Work" section.
- The ICH adapter (16,642 params) on NYU works *because* the backbone is strong (DA-V2). Replacing it with a weaker backbone will require retuning ICH width and learning rate.

---

## 📝 Citation

If R³DC helps your research, please cite:

```bibtex
@inproceedings{mohammad2026rdc,
  title     = {R{\^3}DC: Reliability-Guided Reveal-to-Revise Depth Completion
               for Cross-Domain Sparse Perception},
  author    = {Mohammad, Noor Islam S. and Beyaz{\i}t, Ulu{\u{g}}},
  booktitle = {CVPR 2026 Workshop on 3D Geometry Generation for Scientific Computing},
  year      = {2026},
  url       = {https://openreview.net/forum?id=odj32HFuaj}
}
```

A `CITATION.cff` is provided at the project root so that GitHub's "Cite this repository" widget works out-of-the-box.

---

## 📜 License & acknowledgements

This codebase is released under the [MIT License](LICENSE).

We thank:

- The **CSPN / CSPN++** authors for the spatial-propagation foundations we build on,
- **Depth Anything V2** (Yang et al.) for the strong indoor backbone,
- The **KITTI**, **VisDrone**, and **NYU Depth V2** maintainers for their datasets,
- The reviewers of the CVPR 2026 *3D Geometry Generation for Scientific Computing* workshop for valuable feedback that shaped the honest framing now reflected in this README.

If you have questions, suggestions, or want to contribute — open an issue, send a PR, or reach the authors via OpenReview. We welcome bug reports and especially welcome attempts to break R³DC on new domains.

<div align="center">

*Made with ☕ at Istanbul Technical University.*

</div>
