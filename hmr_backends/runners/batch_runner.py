import numpy as np
import torch
from tqdm import tqdm

from hmr_backends.utils import recursive_to
from hmr_backends.datasets.vitdet_dataset import SequenceVitDetDataset
from hmr_backends.utils.renderer import cam_crop_to_full
from bbox_utils import convert_crop_coords_to_orig_img
from .base import BaseBackendRunner
from hmr_backends.runners.schema import BackendOutputInstance


class BatchBackendRunner(BaseBackendRunner):
    def __init__(self, model, model_cfg, args, device, backend_name):
        super().__init__(model, model_cfg, args, device)
        self.backend_name = backend_name
        self.is_wilor = backend_name == 'wilor'

    def infer(self, inputs):
        entries = []
        for i, inst in enumerate(inputs.instances):
            bbox = inst.bbox.astype(np.float32)
            center = (bbox[2:4] + bbox[0:2]) / 2.0
            scale = self.args.rescale_factor * (bbox[2:4] - bbox[0:2]) / 200.0
            right_val = 1.0 if inst.hand_side == 'right' else 0.0
            entries.append({
                'img_path': inst.img_path,
                'center': center,
                'scale': scale,
                'right': right_val,
                'keypoints': inst.keypoints,
                'instance_idx': i,
            })

        dataset = SequenceVitDetDataset(
            self.model_cfg, entries,
            fp16=self.is_wilor, is_wilor=self.is_wilor,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.args.batch_size,
            shuffle=False, num_workers=0,
        )

        outputs = []
        for batch in tqdm(dataloader, desc=f"Pass 3: Running {self.backend_name.upper()}"):
            batch_idx = batch.pop('instance_idx')
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                out = self.model(batch)

            pred_cam = out['pred_cam']
            multiplier = (2 * batch['right'] - 1)
            pred_cam[:, 1] = multiplier * pred_cam[:, 1]
            box_center = batch['box_center'].float()
            box_size = batch['box_size'].float()
            img_size = batch['img_size'].float()
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            bs = batch['img'].shape[0]

            if self.is_wilor:
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()
            else:
                pred_cam_t_full = torch.zeros(bs, 3, device=pred_cam.device)
                for n in range(bs):
                    cam = pred_cam[n]
                    h, w = img_size[n, 1], img_size[n, 0]
                    focal = scaled_focal_length[n] if scaled_focal_length.ndim > 0 else scaled_focal_length
                    cx, cy = box_center[n]
                    scale = box_size[n]
                    tz = 2 * focal / (scale * cam[0] + 1e-6)
                    tx = cam[1] + tz / focal * (cx - w / 2)
                    ty = cam[2] + tz / focal * (cy - h / 2)
                    pred_cam_t_full[n] = torch.tensor([tx, ty, tz], device=pred_cam.device)
                pred_cam_t_full = pred_cam_t_full.detach().cpu().numpy()

            all_pred_2d = out['pred_keypoints_2d'].detach().cpu().numpy()
            if 'bbox' in batch:
                all_bboxes = batch['bbox'].detach().cpu().numpy()
            else:
                all_bboxes = []
                for n in range(bs):
                    cx = box_center[n, 0].cpu().numpy()
                    cy = box_center[n, 1].cpu().numpy()
                    side = float(box_size[n].cpu().numpy())
                    all_bboxes.append(np.array([cx, cy, side, side]))
                all_bboxes = np.stack(all_bboxes)

            all_pred_2d = self.model_cfg.MODEL.IMAGE_SIZE * (all_pred_2d + 0.5)
            conf = np.ones((all_pred_2d.shape[0], all_pred_2d.shape[1], 1))
            all_pred_2d = np.concatenate((all_pred_2d, conf), axis=-1)
            all_pred_2d = convert_crop_coords_to_orig_img(all_bboxes.copy(), all_pred_2d, self.model_cfg.MODEL.IMAGE_SIZE)
            all_pred_2d[:, :, -1] = 1

            for n in range(bs):
                orig_idx = batch_idx[n].item()
                inst = inputs.instances[orig_idx]
                is_right_val = int(batch['right'][n].detach().cpu().numpy())
                if self.is_wilor:
                    mano_params = {k: v[n].detach().cpu().numpy() for k, v in out['pred_mano_params'].items()}
                else:
                    mano_params = out['pred_mano_params'][n]
                mano_params['is_right'] = is_right_val
                outputs.append(BackendOutputInstance(
                    frame_idx=inst.frame_idx,
                    img_path=inst.img_path,
                    hand_side=inst.hand_side,
                    mano_params=mano_params,
                    cam_trans=pred_cam_t_full[n],
                    pred_vertices=out['pred_vertices'][n].detach().cpu().numpy(),
                    pred_keypoints_2d=all_pred_2d[n],
                    raw_backend_meta={'backend': self.backend_name, 'is_right': is_right_val},
                ))
        return outputs
