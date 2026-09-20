import numpy as np
import torch
from tqdm import tqdm
from .base import BaseBackendRunner
from hmr_backends.runners.schema import BackendOutputInstance


def project_hawor_joints(joints, translation, focal, center, *, is_left):
    """Project HaWoR MANO joints into the original, unflipped image."""
    local = np.asarray(joints, dtype=np.float64).copy()
    if is_left:
        local[:, 0] *= -1
    camera = local + np.asarray(translation, dtype=np.float64).reshape(1, 3)
    valid = np.isfinite(camera).all(axis=1) & (camera[:, 2] > 1e-8)
    keypoints = np.zeros((len(camera), 3), dtype=np.float64)
    keypoints[valid, 0] = focal * camera[valid, 0] / camera[valid, 2] + center[0]
    keypoints[valid, 1] = focal * camera[valid, 1] / camera[valid, 2] + center[1]
    keypoints[valid, 2] = 1.0
    return keypoints


class HaworBackendRunner(BaseBackendRunner):
    def infer(self, inputs):
        outputs = []
        h0, w0 = inputs.image_size
        img_center = [w0 / 2.0, h0 / 2.0]
        img_focal = inputs.img_focal
        segments = inputs.temporal_segments
        if segments is None:
            segments = [
                [inputs.instances_by_key[(idx, side)] for idx in segment]
                for side in ('left', 'right') for segment in inputs.segments_by_hand[side]
            ]
        for seg_instances in tqdm(segments, desc="Pass 3: Running HAWOR tracks"):
            hand_side = seg_instances[0].hand_side
            seg_imgfiles = [inst.img_path for inst in seg_instances]
            seg_boxes = np.stack([inst.bbox for inst in seg_instances], axis=0)
            do_flip = hand_side == 'left'
            results = self.model.inference(seg_imgfiles, seg_boxes, img_focal=img_focal, img_center=img_center, device=str(self.device), do_flip=do_flip)
            pred_rotmat = results['pred_rotmat'].cpu().numpy()
            pred_shape = results['pred_shape'].cpu().numpy()
            pred_trans = results['pred_trans'].cpu().numpy()[:, 0, :]
            for local_i, inst in enumerate(seg_instances):
                rotmat = pred_rotmat[local_i]
                mano_params = {
                    'global_orient': rotmat[[0]],
                    'hand_pose': rotmat[1:],
                    'betas': pred_shape[local_i],
                    'is_right': 1 if hand_side == 'right' else 0,
                }
                mano_output = self.model.mano.query({
                    'pred_rotmat': torch.from_numpy(rotmat).unsqueeze(0).to(self.device),
                    'pred_shape': torch.from_numpy(pred_shape[local_i]).unsqueeze(0).to(self.device),
                })
                joints = None
                if hasattr(mano_output, 'joints'):
                    joints = mano_output.joints[0, :21].detach().cpu().numpy()
                keypoints = (
                    project_hawor_joints(
                        joints,
                        pred_trans[local_i],
                        img_focal,
                        img_center,
                        is_left=(hand_side == 'left'),
                    )
                    if joints is not None else np.zeros((21, 3), dtype=np.float64)
                )
                outputs.append(BackendOutputInstance(
                    frame_idx=inst.frame_idx,
                    img_path=inst.img_path,
                    hand_side=hand_side,
                    mano_params=mano_params,
                    cam_trans=pred_trans[local_i],
                    pred_vertices=mano_output.vertices[0].detach().cpu().numpy(),
                    pred_keypoints_2d=keypoints,
                    pred_joints_3d=joints,
                    raw_backend_meta={
                        'backend': 'hawor',
                        'pred_rotmat': rotmat,
                        'pred_shape': pred_shape[local_i],
                        'img_focal': img_focal,
                        'img_center': img_center,
                        'segment_range': [seg_instances[0].frame_idx, seg_instances[-1].frame_idx],
                        **inst.observation_meta,
                    },
                ))
        return outputs
