"""
MANO forward-pass utilities for inference.

Provides:
  - FACES_NEW / get_extended_faces() — shared face-extension constant
  - run_mano / run_mano_left / run_mano_twohands — MANO forward with axis-angle → rotmat
  - get_mano_faces — return base MANO face indices
"""

import torch
from hmr_backends.models.mano_wrapper import MANO
from hmr_backends.utils.geometry import aa_to_rotmat
import numpy as np
import sys
import os


# ---------------------------------------------------------------------------
# Shared face-extension constant (used by run_mano, run_mano_left, and
# HaWoRRenderAdapter in hmr_backends.utils.render_policy).
# ---------------------------------------------------------------------------
FACES_NEW = np.array([
    [92, 38, 234],
    [234, 38, 239],
    [38, 122, 239],
    [239, 122, 279],
    [122, 118, 279],
    [279, 118, 215],
    [118, 117, 215],
    [215, 117, 214],
    [117, 119, 214],
    [214, 119, 121],
    [119, 120, 121],
    [121, 120, 78],
    [120, 108, 78],
    [78, 108, 79],
])


def get_extended_faces(mano_faces):
    """Return (faces_right, faces_left) with the 14-triangle extension appended.

    Both left and right share the vertex topology; faces_left flips the
    winding order (columns 0 ↔ 2) to get correct normals.
    """
    faces_right = np.concatenate([mano_faces, FACES_NEW], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]
    return faces_right, faces_left


# ---------------------------------------------------------------------------
# MANO instance caching — avoids re-creating MANO on every call.
# ---------------------------------------------------------------------------
_MANO_CACHE = {}


def _resolve_device(device=None):
    """Resolve a torch device from either a torch.device, a string, or None.

    Returns a torch.device.  If *device* is None, defaults to cuda:0 when
    CUDA is available, otherwise cpu.
    """
    if device is None:
        return torch.device('cpu')
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


def _get_right_mano(device=None):
    device = _resolve_device(device)
    key = ('right', str(device))
    if key not in _MANO_CACHE:
        _block_print()
        MANO_cfg = {
            'DATA_DIR': '_DATA/data/',
            'MODEL_PATH': '_DATA/data/mano',
            'GENDER': 'neutral',
            'NUM_HAND_JOINTS': 15,
            'CREATE_BODY_POSE': False,
        }
        mano_cfg = {k.lower(): v for k, v in MANO_cfg.items()}
        mano = MANO(**mano_cfg)
        mano = mano.to(device)
        _MANO_CACHE[key] = mano
        _enable_print()
    return _MANO_CACHE[key]


def _get_left_mano(device=None, fix_shapedirs=True):
    device = _resolve_device(device)
    key = ('left', str(device), fix_shapedirs)
    if key not in _MANO_CACHE:
        _block_print()
        MANO_cfg = {
            'DATA_DIR': '_DATA/data_left/',
            'MODEL_PATH': '_DATA/data_left/mano_left',
            'GENDER': 'neutral',
            'NUM_HAND_JOINTS': 15,
            'CREATE_BODY_POSE': False,
            'is_rhand': False,
        }
        mano_cfg = {k.lower(): v for k, v in MANO_cfg.items()}
        mano = MANO(**mano_cfg)
        mano = mano.to(device)
        # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
        if fix_shapedirs:
            mano.shapedirs[:, 0, :] *= -1
        _MANO_CACHE[key] = mano
        _enable_print()
    return _MANO_CACHE[key]


def _block_print():
    sys.stdout = open(os.devnull, 'w')


def _enable_print():
    sys.stdout = sys.__stdout__


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_mano_faces():
    """Return the base MANO face indices (right hand)."""
    mano = _get_right_mano(device='cpu')
    return mano.faces


def run_mano(trans, root_orient, hand_pose, is_right=None, betas=None, device=None):
    """
    Forward pass of the right-hand MANO model.

    trans : B x T x 3
    root_orient : B x T x 3  (axis-angle)
    hand_pose : B x T x J*3  (axis-angle)
    betas : (optional) B x D
    device : torch device (string or torch.device).  Defaults to cuda:0.
    """
    device = _resolve_device(device)
    mano = _get_right_mano(device=device)
    B, T, _ = root_orient.shape
    NUM_JOINTS = 15
    rotmat_mano_params = {
        'global_orient': aa_to_rotmat(root_orient.reshape(B * T, -1)).view(B * T, 1, 3, 3),
        'hand_pose': aa_to_rotmat(hand_pose.reshape(B * T * NUM_JOINTS, 3)).view(B * T, NUM_JOINTS, 3, 3),
        'betas': betas.reshape(B * T, -1),
        'transl': trans.reshape(B * T, 3),
    }

    mano_output = mano(**{k: v.float().to(device) for k, v in rotmat_mano_params.items()}, pose2rot=False)

    faces_right, faces_left = get_extended_faces(mano.faces)
    faces_n = len(faces_right)

    outputs = {
        "joints": mano_output.joints.reshape(B, T, -1, 3),
        "vertices": mano_output.vertices.reshape(B, T, -1, 3),
    }

    if is_right is not None:
        is_right = (is_right[:, :, 0].cpu().numpy() > 0)
        faces_result = np.zeros((B, T, faces_n, 3))
        faces_right_expanded = np.expand_dims(np.expand_dims(faces_right, axis=0), axis=0)
        faces_left_expanded = np.expand_dims(np.expand_dims(faces_left, axis=0), axis=0)
        faces_result = np.where(is_right[..., np.newaxis, np.newaxis], faces_right_expanded, faces_left_expanded)
        outputs["faces"] = torch.from_numpy(faces_result.astype(np.int32))

    return outputs


def run_mano_rotmat(global_orient, hand_pose, betas, device=None):
    """
    Forward right-hand MANO from saved rotmat parameters.

    global_orient: B x 1 x 3 x 3
    hand_pose: B x 15 x 3 x 3
    betas: B x 10
    """
    device = _resolve_device(device)
    mano = _get_right_mano(device=device)
    global_orient = torch.as_tensor(global_orient, dtype=torch.float32, device=device)
    hand_pose = torch.as_tensor(hand_pose, dtype=torch.float32, device=device)
    betas = torch.as_tensor(betas, dtype=torch.float32, device=device)
    if global_orient.ndim != 4 or global_orient.shape[1:] != (1, 3, 3):
        raise ValueError(f"global_orient must have shape (B, 1, 3, 3), got {tuple(global_orient.shape)}")
    if hand_pose.ndim != 4 or hand_pose.shape[1:] != (15, 3, 3):
        raise ValueError(f"hand_pose must have shape (B, 15, 3, 3), got {tuple(hand_pose.shape)}")
    if betas.ndim != 2 or betas.shape[1] != 10:
        raise ValueError(f"betas must have shape (B, 10), got {tuple(betas.shape)}")
    mano_output = mano(
        global_orient=global_orient,
        hand_pose=hand_pose,
        betas=betas,
        pose2rot=False,
    )
    return {
        "vertices": mano_output.vertices,
        "joints": mano_output.joints,
    }


def run_mano_left(trans, root_orient, hand_pose, is_right=None, betas=None, device=None, fix_shapedirs=True):
    """
    Forward pass of the left-hand MANO model.

    trans : B x T x 3
    root_orient : B x T x 3  (axis-angle)
    hand_pose : B x T x J*3  (axis-angle)
    betas : (optional) B x D
    device : torch device (string or torch.device).  Defaults to cuda:0.
    """
    device = _resolve_device(device)
    mano = _get_left_mano(device=device, fix_shapedirs=fix_shapedirs)
    B, T, _ = root_orient.shape
    NUM_JOINTS = 15
    rotmat_mano_params = {
        'global_orient': aa_to_rotmat(root_orient.reshape(B * T, -1)).view(B * T, 1, 3, 3),
        'hand_pose': aa_to_rotmat(hand_pose.reshape(B * T * NUM_JOINTS, 3)).view(B * T, NUM_JOINTS, 3, 3),
        'betas': betas.reshape(B * T, -1),
        'transl': trans.reshape(B * T, 3),
    }

    mano_output = mano(**{k: v.float().to(device) for k, v in rotmat_mano_params.items()}, pose2rot=False)

    faces_right, faces_left = get_extended_faces(mano.faces)
    faces_n = len(faces_right)

    outputs = {
        "joints": mano_output.joints.reshape(B, T, -1, 3),
        "vertices": mano_output.vertices.reshape(B, T, -1, 3),
    }

    if is_right is not None:
        is_right = (is_right[:, :, 0].cpu().numpy() > 0)
        faces_result = np.zeros((B, T, faces_n, 3))
        faces_right_expanded = np.expand_dims(np.expand_dims(faces_right, axis=0), axis=0)
        faces_left_expanded = np.expand_dims(np.expand_dims(faces_left, axis=0), axis=0)
        faces_result = np.where(is_right[..., np.newaxis, np.newaxis], faces_right_expanded, faces_left_expanded)
        outputs["faces"] = torch.from_numpy(faces_result.astype(np.int32))

    return outputs


def run_mano_twohands(init_trans, init_rot, init_hand_pose, is_right, init_betas, device=None, fix_shapedirs=True):
    outputs_left = run_mano_left(init_trans[0:1], init_rot[0:1], init_hand_pose[0:1], None, init_betas[0:1], device=device, fix_shapedirs=fix_shapedirs)
    outputs_right = run_mano(init_trans[1:2], init_rot[1:2], init_hand_pose[1:2], None, init_betas[1:2], device=device)
    outputs_two = {
        "vertices": torch.cat((outputs_left["vertices"], outputs_right["vertices"]), dim=0),
        "joints": torch.cat((outputs_left["joints"], outputs_right["joints"]), dim=0)
    }
    return outputs_two
