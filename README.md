# EgoHandKit

![EgoHandKit teaser](assets/teaser.gif)

**EgoHandKit** is a toolkit for hand mesh recovery and world-space
MANO reconstruction. It takes an image folder or video, detects hands, runs a
selected HMR backend, recovers per-frame VGGT-Omega cameras, and derives MANO meshes
in the estimated world coordinate system.

The project is centered on **`run.py`** and is designed for egocentric or
hand-centric videos where both image-space overlays and world-space hand
geometry are useful outputs.

Input can be either:

- an image folder, or
- a video file

The pipeline runs in 4 passes:

1. **Detection** — YOLO hand detection, optionally merged with Detectron2 + ViTPose
2. **Cleaning** — temporal bbox cleanup and handedness correction
3. **Mesh Recovery** — run one backend: `hamer`, `htm`, `wilor`, or `hawor`
4. **Omega World Derivation** — recover per-frame Omega cameras and derive MANO world-space results

## Prepare

Place third-party model weights and MANO assets under `_DATA/`. The loaders use
these relative paths as defaults, so keep the directory names unchanged.

Expected layout:

```text
_DATA/
├── data/
│   ├── mano/MANO_RIGHT.pkl
│   └── mano_mean_params.npz
├── hand_det/detector.pt
├── hamer_ckpts/
│   ├── model_config.yaml
│   └── checkpoints/
│       ├── hamer.ckpt
│       └── texture_supervised_hamer_weights.ckpt
├── hawor_ckpts/
│   ├── model_config.yaml
│   └── checkpoints/
│       ├── hawor.ckpt
│       └── infiller.pt
├── wilor_ckpts/
│   ├── model_config.yaml
│   └── wilor_final.ckpt
├── vggt_omega/
│   └── vggt_omega_1b_512.pt
└── vitpose_ckpts/
    ├── configs/ViTPose_huge_wholebody_256x192.py
    └── vitpose+_huge/wholebody.pth
```

Asset sources:

| Asset | Required for | Source |
|---|---|---|
| `data/mano/MANO_RIGHT.pkl` | all HMR backends | [MANO](https://mano.is.tue.mpg.de/) |
| `data/mano_mean_params.npz` | all HMR backends | [HaMeR demo assets](https://github.com/geopavlakos/hamer) |
| `hand_det/detector.pt` | default YOLO hand detection | [WiLoR](https://github.com/rolpotamias/WiLoR) |
| `hamer_ckpts/checkpoints/hamer.ckpt` + `hamer_ckpts/model_config.yaml` | `--backend hamer` | [HaMeR](https://github.com/geopavlakos/hamer) |
| `hamer_ckpts/checkpoints/texture_supervised_hamer_weights.ckpt` | `--backend htm` | [Hand Texture Module](https://github.com/gkarv/Hand-Texture-Module) |
| `wilor_ckpts/wilor_final.ckpt` + `wilor_ckpts/model_config.yaml` | `--backend wilor` | [WiLoR](https://github.com/rolpotamias/WiLoR) |
| `hawor_ckpts/checkpoints/hawor.ckpt` + `hawor_ckpts/checkpoints/infiller.pt` + `hawor_ckpts/model_config.yaml` | `--backend hawor` | [HaWoR](https://github.com/ThunderVVV/HaWoR/) |
| `vggt_omega/vggt_omega_1b_512.pt` | default Omega camera and world derivation | [VGGT-Omega](https://github.com/facebookresearch/vggt-omega) |
| `vitpose_ckpts/configs/ViTPose_huge_wholebody_256x192.py` + `vitpose_ckpts/vitpose+_huge/wholebody.pth` | optional `--use_vitpose` detector merge | [ViTPose](https://github.com/ViTAE-Transformer/ViTPose) |

## Installation

The pip requirements are collected in `requirements.txt`, recommended setup:

```bash
conda create -n egohandkit python=3.10 -y
conda activate egohandkit
pip install -U pip
pip install -r requirements.txt
```

Notes:

- `requirements.txt` defaults to the CUDA 12.8 PyTorch wheel index
  (`torch==2.7.1`, `torchvision==0.22.1`). If your CUDA version is different,
  install the matching PyTorch stack first or adjust the PyTorch lines in
  `requirements.txt`.
- `--use_vitpose` requires the Detectron2 + ViTPose stack. These packages are
  included in `requirements.txt`, but Detectron2 may need to be rebuilt for your
  local CUDA/PyTorch combination.

## Quick Start

```bash
# Image folder input
python run.py --input test_data/images/disk --backend hawor --gpu 0

# Video file input (frames are auto-extracted)
python run.py --input assets/disk.mp4 --backend wilor --gpu 0

# Enable YOLO + Detectron2 + ViTPose merge
python run.py --input test_data/images/disk --backend hamer --use_vitpose

# Switch backend while reusing cached Pass 1 / Pass 2 results
python run.py --input assets/disk.mp4 --backend htm --gpu 0

# Force re-run detection / cleaning
python run.py --input test_data/images/disk --backend hawor --force_detect

```

## Main Options

| Argument | Default | Description |
|---|---|---|
| `--input` | required | Video file or image folder |
| `--backend` | `hawor` | `hamer`, `htm`, `wilor`, `hawor` |
| `--gpu` | `0` | Physical CUDA GPU index. Sets both `CUDA_VISIBLE_DEVICES` and `EGL_DEVICE_ID` before importing torch, then the process uses remapped `cuda:0`. |
| `--fps` | `15` | Output FPS for image-folder input |
| `--render` | `True` | Render mesh overlays |
| `--force_detect` | `False` | Re-run Pass 1 / Pass 2 even if cache exists |
| `--no_clean_bbox` | `False` | Skip Pass 2 temporal bbox cleaning and feed raw Pass 1 detections to Pass 3 |
| `--use_vitpose` | `False` | Merge YOLO with Detectron2 + ViTPose detections |
| `--batch_size` | `48` | Inference batch size (sequence-level, across all frames) |
| `--rescale_factor` | auto | Bbox padding factor |
| `--img_focal` | auto | HaWoR focal: CLI -> `est_focal.txt` -> `600` |
| `--no_omega_world` | `False` | Disable default Pass 4 Omega camera recovery + world-space MANO derivation |
| `--omega_checkpoint` | `_DATA/vggt_omega/vggt_omega_1b_512.pt` | Omega checkpoint path |
| `--omega_image_resolution` | `512` | Omega preprocessing image resolution |
| `--omega_chunk_size` | `auto` | Omega chunk size for long videos |
| `--omega_overlap` | `8` | Overlap frames for Omega chunk alignment |
| `--omega_force` | `False` | Re-run Omega camera recovery even if cache exists |
| `--no_omega_world_vis` | `False` | Disable default Omega world fixed-view 2x2 visualization |

## Key Current Behavior

- **Backend names are unified**: only `hamer`, `htm`, `wilor`, `hawor`
- **GPU control is unified** through `run.py --gpu`, by setting `CUDA_VISIBLE_DEVICES` + `EGL_DEVICE_ID` before importing torch, then using remapped `cuda:0` inside the process
- **Pass 3 batching is sequence-level**: all hand crops across the entire sequence are batched together (up to `--batch_size`), not per-image
- **HaWoR uses its own runner** and focal resolution path
- **Pass 2 can be bypassed** with `--no_clean_bbox`; Pass 3 then consumes raw Pass 1 detections directly
- **Pass 4 Omega world output is enabled by default**: `run.py` estimates per-frame Omega cameras, writes `{name}_{backend}_omega_world.pkl`, and renders a fixed-view 2x2 world visualization; use `--no_omega_world` for MANO-only runs

## Input / Output

### Input

- **Video file**: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
  - frames are extracted to `test_data/images/{name}/`
  - output video uses the source video's native FPS
- **Image folder**
  - images are loaded directly
  - output video uses `--fps`

### Output

Outputs are written to:

```text
test_data/hand_proc/{name}/
```

Typical contents:

```text
pass1_raw.pkl
pass2_cleaned.pkl
{name}_{backend}.pkl
omega_camera.npz
{name}_{backend}_omega_world.pkl
render_{backend}/
render_{backend}.mp4
omega_world_grid_{backend}/
omega_world_grid_{backend}.mp4
```

## Caching

Pass 1 and Pass 2 are cached:

- switching `--backend` normally only re-runs Pass 3 and the backend-specific world pkl
- `omega_camera.npz` is reused across backends unless `--omega_force` is set
- use `--force_detect` to invalidate detection / cleaning cache
- use `--no_clean_bbox` to ignore `pass2_cleaned.pkl` and skip writing a new Pass 2 cache for that run

## Pass 2 Cleaning Summary

The cleaning stage includes:

- overlap removal
- handedness swap correction
- temporal consistency repair
- interpolation over short gaps
- removal of short spurious tracks
- overlap re-check after interpolation

For implementation details, refer to `bbox_utils.py`.

## Project Structure

```text
run.py                  # Main entry point
mesh_recovery.py        # Pass 3 orchestration
bbox_utils.py           # Pass 1 / Pass 2 logic
vitpose_model.py        # Optional ViTPose wrapper
hmr_backends/           # Models, datasets, runners, rendering utilities
  datasets/vitdet_dataset.py   # ViTDetDataset (per-image) + SequenceVitDetDataset (whole-sequence)
  omega/                       # Migrated Omega runtime + Pass 4 camera/world utilities
  runners/batch_runner.py      # Sequence-level batched inference for hamer/htm/wilor
  runners/hawor_runner.py      # Segment-based inference for hawor
```

## Rendering

- **Left hand**: green
- **Right hand**: blue
- Lighting: ambient + 3 Raymond DirectionalLights (PointLights removed for cross-backend brightness consistency)
- **Omega world grid**: default 2x2 video. Top-left is the original overlay; the other three tiles are fixed world-space front / right / top views on a gray background with the current Omega camera shown as a red pyramid.


## Acknowledgement

This work is built on many amazing research works and open-source projects:

- [Dyn-HaMR](https://github.com/ZhengdiYu/Dyn-HaMR)
- [HaMeR](https://github.com/geopavlakos/hamer)
- [HaWoR](https://github.com/ThunderVVV/HaWoR)
- [WiLoR](https://github.com/rolpotamias/WiLoR)
- [VGGT-Omega](https://github.com/facebookresearch/vggt-omega)

Thanks for their excellent works and great contribution.

## Citation

If you find this project useful, please cite:

```bibtex
@misc{wu2026egohandkit,
    title     = {EgoHandKit},
    author    = {Wu, Zijian},
    url       = {https://github.com/Zijian-Wu/EgoHandKit},
    year      = {2026}
}
```
