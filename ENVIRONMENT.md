# Local Environment

Configured and verified on 2026-09-09 in `/home/user/EgoHandKit`.

## Use

```bash
cd /home/user/EgoHandKit
conda activate egohandkit
python run.py --help
python run.py --input assets/disk.mp4 --backend hawor --gpu 0 --batch_size 2 --no_omega_world
```

All listed weights are now available. `--no_omega_world` is optional and skips
camera/world recovery for faster MANO-only runs. HaMeR, WiLoR, and HTM can be
selected with `--backend hamer`, `--backend wilor`, and `--backend htm`.
Run from the repository root because asset paths are relative.

The existing 16-frame smoke sequence also runs with full Omega world recovery:

```bash
python run.py --input test_data/images/environment_smoke --backend hawor --gpu 0 --batch_size 2 --omega_world
```

Only this short sequence has been verified with Omega on the 12 GB GPU;
do not assume the default `auto` chunk size fits long videos in GPU memory.

The Conda environment is at `/home/user/miniconda3/envs/egohandkit`.
It sets `YOLO_CONFIG_DIR=/home/user/EgoHandKit/.cache/ultralytics` on activation
to keep subsequent Ultralytics settings local to this project.

## Verified Stack

| Component | Version |
| --- | --- |
| Python | 3.10.13 |
| PyTorch / torchvision | 2.7.1+cu126 / 0.22.1+cu126 |
| NumPy | 1.26.1 |
| PyTorch Lightning | 2.6.1 |
| GPU | NVIDIA GeForce RTX 4070, 12 GB |
| NVIDIA driver | 550.163.01 |
| Detectron2 | 0.6, revision pinned in requirements.txt |
| ViTPose / MMCV | mmpose 0.24.0 / mmcv 1.3.9 |
| Detectron2 build | CUDA Toolkit 12.2, GCC/G++ 11, architecture 8.9 |

The CUDA 12.6 wheels are an official
[PyTorch 2.7.1 distribution](https://pytorch.org/get-started/previous-versions/).
The existing driver was retained; CUDA 12.x supports
[minor-version compatibility with limitations](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
GPU execution and WiLoR's compiled backbone were tested, rather than relying
only on the version numbers reported by `nvidia-smi`.

The initially installed Conda Python 3.10.21 build had intermittent standard
library errors during PyTorch imports (3 failures in 10 isolated processes).
Changing only the interpreter to 3.10.13 passed 20 consecutive import checks.
This is a local workaround, not a claim about every Python 3.10.21 build.

## Recreate on This Host

These commands require Conda, `uv`, Git, GCC/G++ 11, and `/usr/local/cuda-12.2`.
Use a new environment name to avoid altering the existing verified environment.

```bash
conda create -n egohandkit-rebuild python=3.10.13 pip=25.3 -y
conda activate egohandkit-rebuild
uv pip install --python "$CONDA_PREFIX/bin/python" torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
uv pip install --python "$CONDA_PREFIX/bin/python" setuptools==69.5.1 wheel==0.45.1 Cython==0.29.37 numpy==1.26.1 scipy==1.14.1 ninja==1.11.1.4
CC=/usr/bin/gcc-11 CXX=/usr/bin/g++-11 CUDA_HOME=/usr/local/cuda-12.2 \
  TORCH_CUDA_ARCH_LIST=8.9 MAX_JOBS=4 UV_CONCURRENT_BUILDS=1 \
  uv pip install --python "$CONDA_PREFIX/bin/python" --no-build-isolation --index-strategy unsafe-best-match -r requirements.txt
mkdir -p .cache/ultralytics
export YOLO_CONFIG_DIR="$PWD/.cache/ultralytics"
python -m pip check
python run.py --help
```

PyTorch and the build prerequisites must be installed before the legacy source
packages. `requirements.txt` now pins compatible `iopath` and `yapf`, includes
the detector's `dill` and tracking's `lap` dependencies, and pins ViTPose's Git
revision. Transitive dependencies are not fully locked by this recipe.

## Assets

The following project files are symbolic links to existing local assets. Keep
their targets available; no original checkpoint was modified or duplicated.

| Path under `_DATA/` | Link target |
| --- | --- |
| `data/mano/MANO_RIGHT.pkl` | `/home/user/models/mano/MANO_RIGHT.pkl` |
| `hand_det/detector.pt` | `/home/user/WiLoR/pretrained_models/detector.pt` |
| `hamer_ckpts/checkpoints/hamer.ckpt` | `/home/user/models/hamer/_DATA/hamer_ckpts/checkpoints/hamer.ckpt` |
| `hawor_ckpts/checkpoints/hawor.ckpt` | `/home/user/HaWoR/weights/hawor/checkpoints/hawor.ckpt` |
| `hawor_ckpts/checkpoints/infiller.pt` | `/home/user/HaWoR/weights/hawor/checkpoints/infiller.pt` |
| `wilor_ckpts/wilor_final.ckpt` | `/home/user/WiLoR/pretrained_models/wilor_final.ckpt` |
| `vitpose_ckpts/vitpose+_huge/wholebody.pth` | `/home/user/models/hamer/_DATA/vitpose_ckpts/vitpose+_huge/wholebody.pth` |
| `detectron2/model_final_f05665.pkl` | `/home/user/.torch/iopath_cache/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl` |

Additional weights prepared on 2026-09-09:

- HTM: downloaded the checkpoint linked by the
  [official Hand-Texture-Module repository](https://github.com/gkarv/Hand-Texture-Module)
  to `_DATA/hamer_ckpts/checkpoints/texture_supervised_hamer_weights.ckpt`
  (3,081,578,027 bytes).
- Detectron2: downloaded the
  [official ViTDet checkpoint](https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl)
  (2,765,948,277 bytes) and verified that it is byte-for-byte identical to an
  existing iopath cache entry. The project link reuses that cache, and the
  duplicate temporary download was removed. `run.py` resolves its configured
  URL to this existing cached file without another download.
- Omega: downloaded `vggt_omega_1b_512.pt` from the
  [official gated model repository](https://huggingface.co/facebook/VGGT-Omega)
  after the user's access request was approved. Installed at
  `_DATA/vggt_omega/vggt_omega_1b_512.pt` (4,576,706,117 bytes); both file size
  and SHA-256 matched Hugging Face's authenticated file metadata.

Locally computed SHA-256 digests:

```text
9b9c2858d7caa3aade793faca9e30f32a64ec8d927c97ebf9270823aa02e4954  texture_supervised_hamer_weights.ckpt
8601bc52000c8a87960f3db6a9672596c5e06ce33bc30a3b8f96a96efe42ae60  model_final_f05665.pkl
c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934  vggt_omega_1b_512.pt
```

To download Omega again using the existing authorized local login:

```bash
conda activate egohandkit
cd /home/user/EgoHandKit
env -u ALL_PROXY -u all_proxy HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 python -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id="facebook/VGGT-Omega", filename="vggt_omega_1b_512.pt", local_dir="_DATA/vggt_omega"))'
```

The proxy override is per-command: the host's `ALL_PROXY=socks://...` is not
accepted by this version of httpx; the configured HTTP/HTTPS proxies remain
available. It does not change any global proxy settings.

ViTPose's isolated CUDA inference returned 133 finite keypoints. However, the
existing wholebody checkpoint contains extra `backbone.blocks.*.mlp.experts.*`
keys that the repository's `type='ViT'` config ignores. Confirm the intended
checkpoint/config pairing before relying on its accuracy. The optional Apex
and MMCV deformable-attention warnings did not prevent this inference test.

## Validation Results

- `python -m pip check`: no broken requirements.
- Re-resolving `requirements.txt` with `uv --dry-run`: no changes required.
- Main CLI, HMR/Omega imports, and Detectron2/ViTPose config loading passed.
- CUDA matrix multiplication, torchvision NMS, and Detectron2 rotated NMS passed.
- MANO CUDA forward returned 778 vertices and 21 joints; EGL rendering was nonblank.
- HaWoR, HaMeR, WiLoR, and HTM each produced finite reconstruction data and a
  decodable 16-frame render from the first 16 frames of `assets/disk.mp4`,
  resized to 640x360. All four backends have been tested with cleaned detections.
- Detectron2 ViTDet's checkpoint loaded through the same URL/cache path as
  `run.py`; isolated CUDA inference returned 18 detections with finite boxes.
- HTM's loader ignores the checkpoint's extra training-only `texture_model.*`
  and `recoLoss.*` parameters, as expected for this project's HaMeR-based inference.
- HaWoR with Omega completed Pass 3 and Pass 4 at the default Omega resolution
  of 512 on the 16-frame smoke sequence, reusing the validated detection/cleaning
  caches. All 16 camera extrinsics/intrinsics and all 32 world-space MANO meshes
  were finite. The 1280x720 world-grid video decoded all 16 frames with four
  nonblank views per frame.

Smoke inputs are in `test_data/images/environment_smoke/`; results are in
`test_data/hand_proc/environment_smoke/`, including `render_hawor.mp4`,
`render_hamer.mp4`, `render_wilor.mp4`, `render_htm.mp4`, `omega_camera.npz`,
`environment_smoke_hawor_omega_world.pkl`, and `omega_world_grid_hawor.mp4`.
These generated files are ignored by Git. The complete `--use_vitpose` merged
detection pipeline is not tested; see the checkpoint/config warning above.
