"""MINT-style camera-frame MANO smoother, ported without importing MINT.

This is the production MINT UKF + unscented RTS pass: translation, global
orientation, pose, and shape are smoothed in a representation-safe space
(quaternion/axis-angle for rotations), then converted back to MANO 6D values.
"""

from __future__ import annotations

import numpy as np


_PER_HAND = 109
_MIN_VALID = 4
_ALPHA = 1.0
_BETA_UT = 2.0
_JITTER = 1e-9
DEFAULT_PARAMS = {"q": 0.6, "r": 0.6, "beta": 2.0}


def _rot6d_to_mat(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    a0, a1 = values[..., :3], values[..., 3:]
    n0 = np.linalg.norm(a0, axis=-1, keepdims=True)
    b0 = a0 / np.maximum(n0, 1e-12)
    a1p = a1 - np.sum(b0 * a1, axis=-1, keepdims=True) * b0
    n1 = np.linalg.norm(a1p, axis=-1, keepdims=True)
    b1 = a1p / np.maximum(n1, 1e-12)
    b2 = np.cross(b0, b1)
    matrices = np.stack([b0, b1, b2], axis=-2)
    bad = ((n0[..., 0] < 1e-8) | (n1[..., 0] < 1e-8)
           | ~np.isfinite(matrices).all(axis=(-2, -1)))
    if np.any(bad):
        matrices[bad] = np.eye(3, dtype=np.float64)
    return matrices


def _mat_to_rot6d(matrices: np.ndarray) -> np.ndarray:
    matrices = np.asarray(matrices)
    return matrices[..., :2, :].reshape(*matrices.shape[:-2], 6)


def _quat_sign_continuous(quaternions: np.ndarray) -> np.ndarray:
    result = np.array(quaternions, copy=True)
    for index in range(1, len(result)):
        if np.dot(result[index], result[index - 1]) < 0:
            result[index] = -result[index]
    return result


def _sigma_meas_batch(values: np.ndarray) -> np.ndarray:
    second = values[:, :-2] - 2.0 * values[:, 1:-1] + values[:, 2:]
    median = np.median(second, axis=1, keepdims=True)
    mad = np.median(np.abs(second - median), axis=1)
    return np.maximum(1.4826 * mad / np.sqrt(6.0), 1e-6)


def _sigma_points_batch(mean: np.ndarray, covariance: np.ndarray, lam: float) -> np.ndarray:
    channels, dims = mean.shape
    matrix = (dims + lam) * covariance
    matrix = 0.5 * (matrix + matrix.transpose(0, 2, 1)) + _JITTER * np.eye(dims)
    chol = np.linalg.cholesky(matrix)
    points = np.empty((channels, 2 * dims + 1, dims))
    points[:, 0] = mean
    for index in range(dims):
        column = chol[:, :, index]
        points[:, 1 + index] = mean + column
        points[:, 1 + dims + index] = mean - column
    return points


def _urtss_core(timestamps: np.ndarray, values: np.ndarray,
                observation_noise: np.ndarray, process_noise: np.ndarray) -> np.ndarray:
    channels, frames = values.shape
    if frames < _MIN_VALID:
        return values.copy()
    dims = 2
    lam = _ALPHA ** 2 * (dims + (3 - dims)) - dims
    weights_mean = np.full(2 * dims + 1, 1.0 / (2.0 * (dims + lam)))
    weights_cov = weights_mean.copy()
    weights_mean[0] = lam / (dims + lam)
    weights_cov[0] = weights_mean[0] + 1.0 - _ALPHA ** 2 + _BETA_UT
    identity = np.eye(dims)
    filtered_mean = np.zeros((frames, channels, dims))
    predicted_mean = np.zeros((frames, channels, dims))
    predicted_cov = np.zeros((frames, channels, dims, dims))
    cross_cov = np.zeros((frames, channels, dims, dims))
    current_mean = np.zeros((channels, dims))
    current_mean[:, 0] = values[:, 0]
    current_cov = np.zeros((channels, dims, dims))
    current_cov[:, 0, 0] = np.maximum(observation_noise[:, 0], 1e-12)
    current_cov[:, 1, 1] = np.maximum(np.var(np.diff(values, axis=1), axis=1), 1e-6)
    filtered_mean[0] = current_mean
    for frame in range(1, frames):
        dt = float(timestamps[frame] - timestamps[frame - 1]) or 1.0
        transition = np.array([[1.0, dt], [0.0, 1.0]])
        base_process = np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0], [dt ** 2 / 2.0, dt]])
        process_cov = process_noise[:, None, None] * base_process
        sigma = _sigma_points_batch(current_mean, current_cov, lam)
        propagated = np.einsum("ij,ckj->cki", transition, sigma)
        mean_pred = np.einsum("k,ckj->cj", weights_mean, propagated)
        delta_pred = propagated - mean_pred[:, None, :]
        cov_pred = np.einsum("k,cki,ckj->cij", weights_cov, delta_pred, delta_pred) + process_cov
        cov_pred = 0.5 * (cov_pred + cov_pred.transpose(0, 2, 1)) + _JITTER * identity
        delta_sigma = sigma - current_mean[:, None, :]
        cross_cov[frame] = np.einsum("k,cki,ckj->cij", weights_cov, delta_sigma, delta_pred)
        predicted_mean[frame], predicted_cov[frame] = mean_pred, cov_pred
        update_sigma = _sigma_points_batch(mean_pred, cov_pred, lam)
        observations = update_sigma[:, :, 0]
        observation_mean = np.einsum("k,ck->c", weights_mean, observations)
        delta_observation = observations - observation_mean[:, None]
        innovation = np.einsum("k,ck->c", weights_cov, delta_observation ** 2) + observation_noise[:, frame]
        state_observation = np.einsum("k,cki,ck->ci", weights_cov, update_sigma - mean_pred[:, None, :], delta_observation)
        gain = state_observation / innovation[:, None]
        current_mean = mean_pred + gain * (values[:, frame] - observation_mean)[:, None]
        current_cov = cov_pred - innovation[:, None, None] * np.einsum("ci,cj->cij", gain, gain)
        current_cov = 0.5 * (current_cov + current_cov.transpose(0, 2, 1)) + _JITTER * identity
        filtered_mean[frame] = current_mean
    smoothed = filtered_mean.copy()
    for frame in range(frames - 2, -1, -1):
        gain = np.linalg.solve(predicted_cov[frame + 1].transpose(0, 2, 1), cross_cov[frame + 1].transpose(0, 2, 1)).transpose(0, 2, 1)
        smoothed[frame] = filtered_mean[frame] + np.einsum("cij,cj->ci", gain, smoothed[frame + 1] - predicted_mean[frame + 1])
    return smoothed[:, :, 0].T


def _speed_weight(timestamps: np.ndarray, values: np.ndarray, beta: float) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d
    dt = np.maximum(np.diff(timestamps), 1e-6)
    speed = np.linalg.norm(np.diff(values[:3], axis=1), axis=0) / dt
    speed = uniform_filter1d(np.concatenate([speed[:1], speed]), size=5, mode="nearest")
    median = np.median(speed) + 1e-9
    return (1.0 + beta * np.maximum(speed / median - 1.0, 0.0)) ** 2


def _smooth_channels(timestamps: np.ndarray, values: np.ndarray, *, q: float, r: float, beta: float) -> np.ndarray:
    sigma = _sigma_meas_batch(values)
    weight = _speed_weight(timestamps, values, beta)
    return _urtss_core(timestamps, values, (r * sigma)[:, None] ** 2 * weight[None, :], (q * sigma) ** 2)


def smooth_hand_sequence(hand: np.ndarray, frame_indices: np.ndarray | None = None, *,
                         q: float = 0.6, r: float = 0.6, beta: float = 2.0) -> np.ndarray:
    """Smooth one physical track of MANO parameters shaped ``[T,109]``."""
    from scipy.spatial.transform import Rotation
    source = np.asarray(hand)
    if source.ndim != 2 or source.shape[1] != _PER_HAND:
        raise ValueError(f"Expected [T,109] MANO sequence, got {source.shape}")
    if len(source) < _MIN_VALID:
        return source.copy()
    timestamps = np.arange(len(source), dtype=float) if frame_indices is None else np.asarray(frame_indices, dtype=float)
    if timestamps.shape != (len(source),) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("frame_indices must be strictly increasing")
    result = source.astype(np.float64, copy=True)
    orientation = Rotation.from_matrix(_rot6d_to_mat(result[:, 3:9])).as_quat()
    orientation = _quat_sign_continuous(orientation)
    pose_matrices = _rot6d_to_mat(result[:, 9:99].reshape(-1, 15, 6))
    pose_rotvec = Rotation.from_matrix(pose_matrices.reshape(-1, 3, 3)).as_rotvec().reshape(-1, 45)
    channels = np.vstack([result[:, :3].T, orientation.T, pose_rotvec.T, result[:, 99:109].T])
    smoothed = _smooth_channels(timestamps, channels, q=float(q), r=float(r), beta=float(beta))
    offset = 0
    result[:, :3] = smoothed[offset:offset + 3].T; offset += 3
    quat = smoothed[offset:offset + 4].T; offset += 4
    quat /= np.maximum(np.linalg.norm(quat, axis=1, keepdims=True), 1e-8)
    result[:, 3:9] = _mat_to_rot6d(Rotation.from_quat(quat).as_matrix())
    pose = smoothed[offset:offset + 45].T; offset += 45
    result[:, 9:99] = _mat_to_rot6d(Rotation.from_rotvec(pose.reshape(-1, 3)).as_matrix()).reshape(-1, 90)
    result[:, 99:109] = smoothed[offset:offset + 10].T
    return result.astype(source.dtype, copy=False)


def smooth_camera_joint_sequence(joints, frame_indices, *, q=0.6, r=0.6, beta=2.0):
    """Smooth a continuous final track as wrist plus root-relative joints."""
    source = np.asarray(joints)
    times = np.asarray(frame_indices, dtype=np.float64)
    if source.shape != (len(times), 21, 3) or not np.isfinite(source).all():
        raise ValueError('Expected finite [T,21,3] camera joints')
    if np.any(np.diff(times) <= 0):
        raise ValueError('frame_indices must be strictly increasing')
    if not np.isfinite([q, r, beta]).all() or q <= 0 or r <= 0 or beta < 0:
        raise ValueError('Invalid final joint smoother parameters')
    if len(source) < _MIN_VALID:
        return source.copy()
    relative = source[:, 1:] - source[:, :1]
    channels = np.concatenate([source[:, 0], relative.reshape(len(source), -1)], axis=1).T
    filtered = _smooth_channels(times, channels, q=q, r=r, beta=beta).T
    result = np.empty_like(source)
    result[:, 0] = filtered[:, :3]
    result[:, 1:] = filtered[:, 3:].reshape(-1, 20, 3) + result[:, :1]
    # A numerical failure must not turn a valid exported hand into a missing one.
    if not np.isfinite(result).all() or np.any(result[:, :, 2] <= 0):
        raise ValueError('Final joint smoothing produced invalid camera geometry')
    return result
