import os
from pathlib import Path
import torch

from .mano_wrapper import MANO
from .hamer import HAMER
from .wilor import WiLoR
from .hawor import HAWOR
from .discriminator import Discriminator

from ..utils.download import cache_url
from ..configs import get_config


def download_models(folder=None):
    """Download checkpoints and files for running inference.
    """

    os.makedirs(folder, exist_ok=True)
    download_files = {
        "hamer_demo_data.tar.gz"      : ["https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz", folder],
    }

    for file_name, url in download_files.items():
        output_path = os.path.join(url[1], file_name)
        if not os.path.exists(os.path.join(folder, '_DATA', 'hamer_ckpts')):
            print("Downloading file: " + file_name, "to ", output_path)
            output = cache_url(url[0], output_path)
            assert os.path.exists(output_path), f"{output} does not exist"

            # if ends with tar.gz, tar -xzf
            if file_name.endswith(".tar.gz"):
                print("Extracting file: " + file_name, 'to ', folder)
                os.system("tar -xvf " + output_path + f" -C {folder}")


def load_hamer(root=None, checkpoint_name='hamer.ckpt'):
    checkpoint_path = os.path.join(root, f'_DATA/hamer_ckpts/checkpoints/{checkpoint_name}')
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)

    # Defrost the config to allow modifications
    model_cfg.defrost()

    # Update MANO configuration
    model_cfg.MANO.DATA_DIR = os.path.join(root, '_DATA/data/')
    model_cfg.MANO.MEAN_PARAMS = os.path.join(root, '_DATA/data/mano_mean_params.npz')
    model_cfg.MANO.MODEL_PATH = os.path.join(root, '_DATA/data/mano')
    model_cfg.EXTRA.FOCAL_LENGTH = 5000.0

    # Override some config values, to crop bbox correctly
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]

    # Update config to be compatible with demo
    if 'PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE:
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')

    # Freeze the config after modifications
    model_cfg.freeze()

    model = HAMER.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, weights_only=False)
    return model, model_cfg


def load_htm(htm_dir, checkpoint_name='texture_supervised_hamer_weights.ckpt'):
    """Load HTM model (architecturally identical to HaMeR, texture-supervised weights).

    Args:
        htm_dir: Path to the HTM directory (e.g. third-party/htm)
        checkpoint_name: Name of the checkpoint file
    Returns:
        model: HAMER model with HTM weights
        model_cfg: Model configuration
    """
    checkpoint_path = os.path.join(htm_dir, f'_DATA/hamer_ckpts/checkpoints/{checkpoint_name}')
    model_cfg = str(Path(checkpoint_path).parent.parent / 'model_config.yaml')
    model_cfg = get_config(model_cfg, update_cachedir=True)

    model_cfg.defrost()

    model_cfg.MANO.DATA_DIR = os.path.join(htm_dir, '_DATA/data/')
    model_cfg.MANO.MEAN_PARAMS = os.path.join(htm_dir, '_DATA/data/mano_mean_params.npz')
    model_cfg.MANO.MODEL_PATH = os.path.join(htm_dir, '_DATA/data/mano')
    model_cfg.EXTRA.FOCAL_LENGTH = 5000.0

    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        assert model_cfg.MODEL.IMAGE_SIZE == 256
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]

    if 'PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE:
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')

    model_cfg.freeze()

    model = HAMER.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, weights_only=False)
    return model, model_cfg


def load_wilor(root=None):
    """Load WiLoR model from pretrained weights under root/_DATA/.

    Args:
        root: Path to the project root directory
    Returns:
        model: WiLoR model
        model_cfg: Model configuration
    """
    checkpoint_path = os.path.join(root, '_DATA', 'wilor_ckpts', 'wilor_final.ckpt')
    cfg_path = os.path.join(root, '_DATA', 'wilor_ckpts', 'model_config.yaml')

    print('Loading WiLoR from', checkpoint_path)
    model_cfg = get_config(cfg_path, update_cachedir=False)

    model_cfg.defrost()

    # Use the WiLoR-specific backbone type
    model_cfg.MODEL.BACKBONE.TYPE = 'vit_wilor'

    # Override bbox shape for ViT
    if 'BBOX_SHAPE' not in model_cfg.MODEL:
        assert model_cfg.MODEL.IMAGE_SIZE == 256, f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]

    # Remove pretrained weights (loaded from checkpoint)
    if 'PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE:
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')

    # Set MANO data paths (shared _DATA/data/ directory)
    model_cfg.MANO.DATA_DIR = os.path.join(root, '_DATA', 'data')
    model_cfg.MANO.MODEL_PATH = os.path.join(root, '_DATA', 'data', 'mano')
    model_cfg.MANO.MEAN_PARAMS = os.path.join(root, '_DATA', 'data', 'mano_mean_params.npz')

    model_cfg.freeze()

    model = WiLoR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)

    # Follow the official WiLoR demo fast-mode behavior by default in this repo.
    # The dataset path already feeds FP16 inputs for WiLoR, so the model side must
    # also be configured consistently.
    torch.set_float32_matmul_precision('high')
    model = model.half()
    if hasattr(model, 'backbone') and model.backbone is not None:
        model.backbone.skip_blocks = True
        try:
            model.backbone = torch.compile(model.backbone)
        except Exception as e:
            print(f'WARNING: torch.compile(model.backbone) failed, fallback to eager mode: {e}')

    return model, model_cfg


def load_hawor(root=None, checkpoint_name='hawor.ckpt'):
    """Load HaWoR model from pretrained weights under root/_DATA/.

    Args:
        root: Path to the project root directory
        checkpoint_name: Name of the checkpoint file
    Returns:
        model: HAWOR model
        model_cfg: Model configuration
    """
    checkpoint_path = os.path.join(root, '_DATA', 'hawor_ckpts', 'checkpoints', checkpoint_name)
    cfg_path = os.path.join(root, '_DATA', 'hawor_ckpts', 'model_config.yaml')
    model_cfg = get_config(cfg_path, update_cachedir=True)

    model_cfg.defrost()
    model_cfg.MANO.DATA_DIR = os.path.join(root, '_DATA', 'data')
    model_cfg.MANO.MODEL_PATH = os.path.join(root, '_DATA', 'data', 'mano')
    model_cfg.MANO.MEAN_PARAMS = os.path.join(root, '_DATA', 'data', 'mano_mean_params.npz')
    if (model_cfg.MODEL.BACKBONE.TYPE == 'vit') and ('BBOX_SHAPE' not in model_cfg.MODEL):
        assert model_cfg.MODEL.IMAGE_SIZE == 256
        model_cfg.MODEL.BBOX_SHAPE = [192, 256]
    if 'PRETRAINED_WEIGHTS' in model_cfg.MODEL.BACKBONE:
        model_cfg.MODEL.BACKBONE.pop('PRETRAINED_WEIGHTS')
    model_cfg.freeze()

    model = HAWOR.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg, weights_only=False)
    return model, model_cfg


# ---------------------------------------------------------------------------
# Backend loading unified API (moved from hand_recon/backends/loader.py)
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class BackendBundle:
    backend_name: str
    model: object
    model_cfg: object
    render_cfg: object


def load_backend(backend_name: str, repo_root: str, args):
    """Load a backend model and return a BackendBundle.

    Supported backends: hamer, htm, wilor, hawor.
    """
    if backend_name == 'wilor':
        model, model_cfg = load_wilor(repo_root)
        return BackendBundle(backend_name, model, model_cfg, model_cfg)

    if backend_name == 'htm':
        model, model_cfg = load_htm(repo_root)
        return BackendBundle(backend_name, model, model_cfg, model_cfg)

    if backend_name == 'hawor':
        model, model_cfg = load_hawor(repo_root)
        render_cfg = type('RenderCfg', (), {
            'EXTRA': type('Extra', (), {'FOCAL_LENGTH': args.img_focal or 600.0}),
            'MODEL': type('Model', (), {'IMAGE_SIZE': 256}),
        })
        return BackendBundle(backend_name, model, model_cfg, render_cfg)

    # Default: hamer
    download_models(repo_root)
    model, model_cfg = load_hamer(repo_root)
    return BackendBundle(backend_name, model, model_cfg, model_cfg)
