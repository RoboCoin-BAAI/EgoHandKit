from typing import Dict

import cv2
import numpy as np
from skimage.filters import gaussian
from yacs.config import CfgNode
import torch

from .utils import (convert_cvimg_to_tensor,
                    expand_to_aspect_ratio,
                    generate_image_patch_cv2)

DEFAULT_MEAN = 255. * np.array([0.485, 0.456, 0.406])
DEFAULT_STD = 255. * np.array([0.229, 0.224, 0.225])

class ViTDetDataset(torch.utils.data.Dataset):

    def __init__(self,
                 cfg: CfgNode,
                 img_cv2: np.array,
                 boxes: np.array,
                 right: np.array,
                 vit_keypoints: np.array = None,
                 rescale_factor=2.5,
                 train: bool = False,
                 fp16: bool = False,
                 **kwargs):
        super().__init__()
        self.cfg = cfg
        self.img_cv2 = img_cv2
        self.boxes = boxes
        self.fp16 = fp16

        assert train == False, "ViTDetDataset is only for inference"
        self.train = train
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(self.cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(self.cfg.MODEL.IMAGE_STD)

        # Preprocess annotations
        boxes = boxes.astype(np.float32)
        self.center = (boxes[:, 2:4] + boxes[:, 0:2]) / 2.0
        self.scale = rescale_factor * (boxes[:, 2:4] - boxes[:, 0:2]) / 200.0
        self.personid = np.arange(len(boxes), dtype=np.int32)
        self.right = right.astype(np.float32)
        self.vit_keypoints = vit_keypoints.astype(np.float32) if vit_keypoints is not None else None
        self.height, self.weight = boxes[:, 3]-boxes[:, 1], boxes[:, 2]-boxes[:, 0]

    def __len__(self) -> int:
        return len(self.personid)

    def __getitem__(self, idx: int) -> Dict[str, np.array]:

        center = self.center[idx].copy()
        center_x = center[0]
        center_y = center[1]

        scale = self.scale[idx]
        BBOX_SHAPE = self.cfg.MODEL.get('BBOX_SHAPE', None)
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=BBOX_SHAPE).max()
        # print('bbox_size/scale: ', bbox_size, '...', scale)

        patch_width = patch_height = self.img_size

        right = self.right[idx].copy()
        flip = right == 0

        # 3. generate image patch
        # if use_skimage_antialias:
        cvimg = self.img_cv2.copy()
        if True:
            # Blur image to avoid aliasing artifacts
            downsampling_factor = ((bbox_size*1.0) / patch_width)
            # print(f'{downsampling_factor=}')
            downsampling_factor = downsampling_factor / 2.0
            if downsampling_factor > 1.1:
                cvimg  = gaussian(cvimg, sigma=(downsampling_factor-1)/2, channel_axis=2, preserve_range=True)


        img_patch_cv, trans, inv_trans = generate_image_patch_cv2(cvimg,
                                                    center_x, center_y,
                                                    bbox_size, bbox_size,
                                                    patch_width, patch_height,
                                                    flip, 1.0, 0,
                                                    border_mode=cv2.BORDER_CONSTANT)
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)

        # apply normalization
        for n_c in range(min(self.img_cv2.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        if self.fp16:
            img_patch = torch.from_numpy(img_patch).half()

        item = {
            'img': img_patch,
            'personid': int(self.personid[idx]),
        }
        item['box_center'] = self.center[idx].copy()
        item['box_size'] = bbox_size
        item['img_size'] = 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]])
        item['right'] = self.right[idx].copy()

        # Include keypoints and bbox info when available (HaMeR mode)
        if self.vit_keypoints is not None:
            item['img_patch'] = img_patch_cv.copy()
            item['2d'] = self.vit_keypoints[idx].copy()
            item['inv_trans'] = inv_trans.copy()
            item['bbox'] = np.array([center[0], center[1], bbox_size, bbox_size])

        return item


class SequenceVitDetDataset(torch.utils.data.Dataset):
    """Dataset spanning all hand instances across a full image sequence.

    Unlike ViTDetDataset (which takes a single pre-loaded image), this class
    stores per-instance metadata and lazily loads images in __getitem__.
    The crop/normalize pipeline is identical to ViTDetDataset.
    """

    def __init__(self, cfg, entries, fp16=False, is_wilor=False):
        """
        Args:
            cfg: Model config (CfgNode) with MODEL.IMAGE_SIZE, IMAGE_MEAN, IMAGE_STD, BBOX_SHAPE.
            entries: List of dicts, each with keys:
                img_path (str), center (ndarray[2]), scale (ndarray[2]),
                right (float), keypoints (ndarray|None), instance_idx (int).
            fp16: Whether to output half-precision image tensors.
            is_wilor: If True, skip keypoints in output (WiLoR mode).
        """
        super().__init__()
        self.cfg = cfg
        self.entries = entries
        self.fp16 = fp16
        self.is_wilor = is_wilor
        self.img_size = cfg.MODEL.IMAGE_SIZE
        self.mean = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.std = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.BBOX_SHAPE = cfg.MODEL.get('BBOX_SHAPE', None)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        center = entry['center'].copy()
        scale = entry['scale']
        right = entry['right']
        flip = right == 0

        bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=self.BBOX_SHAPE).max()
        patch_width = patch_height = self.img_size

        cvimg = cv2.imread(entry['img_path'])
        downsampling_factor = ((bbox_size * 1.0) / patch_width) / 2.0
        if downsampling_factor > 1.1:
            cvimg = gaussian(cvimg, sigma=(downsampling_factor - 1) / 2, channel_axis=2, preserve_range=True)

        img_patch_cv, trans, inv_trans = generate_image_patch_cv2(
            cvimg, center[0], center[1],
            bbox_size, bbox_size,
            patch_width, patch_height,
            flip, 1.0, 0,
            border_mode=cv2.BORDER_CONSTANT,
        )
        img_patch_cv = img_patch_cv[:, :, ::-1]
        img_patch = convert_cvimg_to_tensor(img_patch_cv)

        for n_c in range(min(cvimg.shape[2], 3)):
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - self.mean[n_c]) / self.std[n_c]

        if self.fp16:
            img_patch = torch.from_numpy(img_patch).half()

        item = {
            'img': img_patch,
            'personid': idx,
            'box_center': center,
            'box_size': bbox_size,
            'img_size': 1.0 * np.array([cvimg.shape[1], cvimg.shape[0]]),
            'right': right,
            'instance_idx': entry['instance_idx'],
        }

        if not self.is_wilor:
            kp = entry['keypoints']
            if kp is None:
                kp = np.zeros((21, 3), dtype=np.float32)
                kp[:, 2] = 0.5
            item['img_patch'] = img_patch_cv.copy()
            item['2d'] = kp.astype(np.float32)
            item['inv_trans'] = inv_trans.copy()
            item['bbox'] = np.array([center[0], center[1], bbox_size, bbox_size])

        return item
