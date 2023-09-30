import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.transforms import functional as F
import math
import onepose.models.vitpose as vitpose
import importlib.util
import os
import warnings
import pathlib
import hashlib
import urllib.request
from tqdm import tqdm
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Sequence, Union, Tuple

def _get_max_preds(heatmaps):
    """Get keypoint predictions from score maps.

    Note:
        batch_size: N
        num_keypoints: K
        heatmap height: H
        heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.

    Returns:
        tuple: A tuple containing aggregated results.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    assert isinstance(heatmaps,
                      np.ndarray), ('heatmaps should be numpy.ndarray')
    assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

    N, K, _, W = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape((N, K, -1))
    idx = np.argmax(heatmaps_reshaped, 2).reshape((N, K, 1))
    maxvals = np.amax(heatmaps_reshaped, 2).reshape((N, K, 1))

    preds = np.tile(idx, (1, 1, 2)).astype(np.float32)
    preds[:, :, 0] = preds[:, :, 0] % W
    preds[:, :, 1] = preds[:, :, 1] // W

    preds = np.where(np.tile(maxvals, (1, 1, 2)) > 0.0, preds, -1)
    return preds, maxvals

def _gaussian_blur(heatmaps, kernel=11):
    """Modulate heatmap distribution with Gaussian.
     sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
     sigma~=3 if k=17
     sigma=2 if k=11;
     sigma~=1.5 if k=7;
     sigma~=1 if k=3;

    Note:
        - batch_size: N
        - num_keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.

    Returns:
        np.ndarray ([N, K, H, W]): Modulated heatmap distribution.
    """
    assert kernel % 2 == 1

    border = (kernel - 1) // 2
    batch_size = heatmaps.shape[0]
    num_joints = heatmaps.shape[1]
    height = heatmaps.shape[2]
    width = heatmaps.shape[3]
    for i in range(batch_size):
        for j in range(num_joints):
            origin_max = np.max(heatmaps[i, j])
            dr = np.zeros((height + 2 * border, width + 2 * border),
                          dtype=np.float32)
            dr[border:-border, border:-border] = heatmaps[i, j].copy()
            dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
            heatmaps[i, j] = dr[border:-border, border:-border].copy()
            heatmaps[i, j] *= origin_max / np.max(heatmaps[i, j])
    return heatmaps

def post_dark_udp(coords, batch_heatmaps, kernel=3):
    """DARK post-pocessing. Implemented by udp. Paper ref: Huang et al. The
    Devil is in the Details: Delving into Unbiased Data Processing for Human
    Pose Estimation (CVPR 2020). Zhang et al. Distribution-Aware Coordinate
    Representation for Human Pose Estimation (CVPR 2020).

    Note:
        - batch size: B
        - num keypoints: K
        - num persons: N
        - height of heatmaps: H
        - width of heatmaps: W

        B=1 for bottom_up paradigm where all persons share the same heatmap.
        B=N for top_down paradigm where each person has its own heatmaps.

    Args:
        coords (np.ndarray[N, K, 2]): Initial coordinates of human pose.
        batch_heatmaps (np.ndarray[B, K, H, W]): batch_heatmaps
        kernel (int): Gaussian kernel size (K) for modulation.

    Returns:
        np.ndarray([N, K, 2]): Refined coordinates.
    """
    if not isinstance(batch_heatmaps, np.ndarray):
        batch_heatmaps = batch_heatmaps.cpu().numpy()
    B, K, H, W = batch_heatmaps.shape
    N = coords.shape[0]
    assert (B == 1 or B == N)
    for heatmaps in batch_heatmaps:
        for heatmap in heatmaps:
            cv2.GaussianBlur(heatmap, (kernel, kernel), 0, heatmap)
    np.clip(batch_heatmaps, 0.001, 50, batch_heatmaps)
    np.log(batch_heatmaps, batch_heatmaps)

    batch_heatmaps_pad = np.pad(
        batch_heatmaps, ((0, 0), (0, 0), (1, 1), (1, 1)),
        mode='edge').flatten()

    index = coords[..., 0] + 1 + (coords[..., 1] + 1) * (W + 2)
    index += (W + 2) * (H + 2) * np.arange(0, B * K).reshape(-1, K)
    index = index.astype(int).reshape(-1, 1)
    i_ = batch_heatmaps_pad[index]
    ix1 = batch_heatmaps_pad[index + 1]
    iy1 = batch_heatmaps_pad[index + W + 2]
    ix1y1 = batch_heatmaps_pad[index + W + 3]
    ix1_y1_ = batch_heatmaps_pad[index - W - 3]
    ix1_ = batch_heatmaps_pad[index - 1]
    iy1_ = batch_heatmaps_pad[index - 2 - W]

    dx = 0.5 * (ix1 - ix1_)
    dy = 0.5 * (iy1 - iy1_)
    derivative = np.concatenate([dx, dy], axis=1)
    derivative = derivative.reshape(N, K, 2, 1)
    dxx = ix1 - 2 * i_ + ix1_
    dyy = iy1 - 2 * i_ + iy1_
    dxy = 0.5 * (ix1y1 - ix1 - iy1 + i_ + i_ - ix1_ - iy1_ + ix1_y1_)
    hessian = np.concatenate([dxx, dxy, dxy, dyy], axis=1)
    hessian = hessian.reshape(N, K, 2, 2)
    hessian = np.linalg.inv(hessian + np.finfo(np.float32).eps * np.eye(2))
    coords -= np.einsum('ijmn,ijnk->ijmk', hessian, derivative).squeeze()
    return coords

def transform_preds(coords, center, scale, output_size, use_udp=False):
    """Get final keypoint predictions from heatmaps and apply scaling and
    translation to map them back to the image.

    Note:
        num_keypoints: K

    Args:
        coords (np.ndarray[K, ndims]):

            * If ndims=2, corrds are predicted keypoint location.
            * If ndims=4, corrds are composed of (x, y, scores, tags)
            * If ndims=5, corrds are composed of (x, y, scores, tags,
              flipped_tags)

        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        use_udp (bool): Use unbiased data processing

    Returns:
        np.ndarray: Predicted coordinates in the images.
    """
    assert coords.shape[1] in (2, 4, 5)
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2

    # Recover the scale which is normalized by a factor of 200.
    scale = scale * 200.0
    
    if use_udp:
        scale_x = scale[0] / (output_size[0] - 1.0)
        scale_y = scale[1] / (output_size[1] - 1.0)
    else:
        scale_x = scale[0] / output_size[0]
        scale_y = scale[1] / output_size[1]

    target_coords = np.ones_like(coords)
    target_coords[:, 0] = coords[:, 0] * scale_x + center[0] - scale[0] * 0.5
    target_coords[:, 1] = coords[:, 1] * scale_y + center[1] - scale[1] * 0.5

    return target_coords

def keypoints_from_heatmaps(heatmaps,
                            center,
                            scale,
                            unbiased=False,
                            post_process='default',
                            kernel=11,
                            valid_radius_factor=0.0546875,
                            use_udp=False,
                            target_type='GaussianHeatmap'):
    """Get final keypoint predictions from heatmaps and transform them back to
    the image.

    Note:
        - batch size: N
        - num keypoints: K
        - heatmap height: H
        - heatmap width: W

    Args:
        heatmaps (np.ndarray[N, K, H, W]): model predicted heatmaps.
        center (np.ndarray[N, 2]): Center of the bounding box (x, y).
        scale (np.ndarray[N, 2]): Scale of the bounding box
            wrt height/width.
        post_process (str/None): Choice of methods to post-process
            heatmaps. Currently supported: None, 'default', 'unbiased',
            'megvii'.
        unbiased (bool): Option to use unbiased decoding. Mutually
            exclusive with megvii.
            Note: this arg is deprecated and unbiased=True can be replaced
            by post_process='unbiased'
            Paper ref: Zhang et al. Distribution-Aware Coordinate
            Representation for Human Pose Estimation (CVPR 2020).
        kernel (int): Gaussian kernel size (K) for modulation, which should
            match the heatmap gaussian sigma when training.
            K=17 for sigma=3 and k=11 for sigma=2.
        valid_radius_factor (float): The radius factor of the positive area
            in classification heatmap for UDP.
        use_udp (bool): Use unbiased data processing.
        target_type (str): 'GaussianHeatmap' or 'CombinedTarget'.
            GaussianHeatmap: Classification target with gaussian distribution.
            CombinedTarget: The combination of classification target
            (response map) and regression target (offset map).
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).

    Returns:
        tuple: A tuple containing keypoint predictions and scores.

        - preds (np.ndarray[N, K, 2]): Predicted keypoint location in images.
        - maxvals (np.ndarray[N, K, 1]): Scores (confidence) of the keypoints.
    """
    # Avoid being affected
    heatmaps = heatmaps.copy()

    # detect conflicts
    if unbiased:
        assert post_process not in [False, None, 'megvii']
    if post_process in ['megvii', 'unbiased']:
        assert kernel > 0
    if use_udp:
        assert not post_process == 'megvii'

    # normalize configs
    if post_process is False:
        warnings.warn(
            'post_process=False is deprecated, '
            'please use post_process=None instead', DeprecationWarning)
        post_process = None
    elif post_process is True:
        if unbiased is True:
            warnings.warn(
                'post_process=True, unbiased=True is deprecated,'
                " please use post_process='unbiased' instead",
                DeprecationWarning)
            post_process = 'unbiased'
        else:
            warnings.warn(
                'post_process=True, unbiased=False is deprecated, '
                "please use post_process='default' instead",
                DeprecationWarning)
            post_process = 'default'
    elif post_process == 'default':
        if unbiased is True:
            warnings.warn(
                'unbiased=True is deprecated, please use '
                "post_process='unbiased' instead", DeprecationWarning)
            post_process = 'unbiased'

    # start processing
    if post_process == 'megvii':
        heatmaps = _gaussian_blur(heatmaps, kernel=kernel)

    N, K, H, W = heatmaps.shape
    if use_udp:
        if target_type.lower() == 'GaussianHeatMap'.lower():
            preds, maxvals = _get_max_preds(heatmaps)
            preds = post_dark_udp(preds, heatmaps, kernel=kernel)
        elif target_type.lower() == 'CombinedTarget'.lower():
            for person_heatmaps in heatmaps:
                for i, heatmap in enumerate(person_heatmaps):
                    kt = 2 * kernel + 1 if i % 3 == 0 else kernel
                    cv2.GaussianBlur(heatmap, (kt, kt), 0, heatmap)
            # valid radius is in direct proportion to the height of heatmap.
            valid_radius = valid_radius_factor * H
            offset_x = heatmaps[:, 1::3, :].flatten() * valid_radius
            offset_y = heatmaps[:, 2::3, :].flatten() * valid_radius
            heatmaps = heatmaps[:, ::3, :]
            preds, maxvals = _get_max_preds(heatmaps)
            index = preds[..., 0] + preds[..., 1] * W
            index += W * H * np.arange(0, N * K / 3)
            index = index.astype(int).reshape(N, K // 3, 1)
            preds += np.concatenate((offset_x[index], offset_y[index]), axis=2)
        else:
            raise ValueError('target_type should be either '
                             "'GaussianHeatmap' or 'CombinedTarget'")
    # else:
    #     preds, maxvals = _get_max_preds(heatmaps)
    #     if post_process == 'unbiased':  # alleviate biased coordinate
    #         # apply Gaussian distribution modulation.
    #         heatmaps = np.log(
    #             np.maximum(_gaussian_blur(heatmaps, kernel), 1e-10))
    #         for n in range(N):
    #             for k in range(K):
    #                 preds[n][k] = _taylor(heatmaps[n][k], preds[n][k])
    #     elif post_process is not None:
    #         # add +/-0.25 shift to the predicted locations for higher acc.
    #         for n in range(N):
    #             for k in range(K):
    #                 heatmap = heatmaps[n][k]
    #                 px = int(preds[n][k][0])
    #                 py = int(preds[n][k][1])
    #                 if 1 < px < W - 1 and 1 < py < H - 1:
    #                     diff = np.array([
    #                         heatmap[py][px + 1] - heatmap[py][px - 1],
    #                         heatmap[py + 1][px] - heatmap[py - 1][px]
    #                     ])
    #                     preds[n][k] += np.sign(diff) * .25
    #                     if post_process == 'megvii':
    #                         preds[n][k] += 0.5

    # Transform back to the image
    for i in range(N):
        preds[i] = transform_preds(
            preds[i], center[i], scale[i], [W, H], use_udp=use_udp)

    if post_process == 'megvii':
        maxvals = maxvals / 255.0 + 0.5

    return preds, maxvals

def read_cfg(cfg):
    spec = importlib.util.spec_from_file_location("config", cfg)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module


def get_warp_matrix(theta, size_input, size_dst, size_target):
    """Calculate the transformation matrix under the constraint of unbiased.
    Paper ref: Huang et al. The Devil is in the Details: Delving into Unbiased
    Data Processing for Human Pose Estimation (CVPR 2020).

    Args:
        theta (float): Rotation angle in degrees.
        size_input (np.ndarray): Size of input image [w, h].
        size_dst (np.ndarray): Size of output image [w, h].
        size_target (np.ndarray): Size of ROI in input plane [w, h].

    Returns:
        np.ndarray: A matrix for transformation.
    """
    theta = np.deg2rad(theta)
    matrix = np.zeros((2, 3), dtype=np.float32)
    scale_x = size_dst[0] / size_target[0]
    scale_y = size_dst[1] / size_target[1]
    matrix[0, 0] = math.cos(theta) * scale_x
    matrix[0, 1] = -math.sin(theta) * scale_x
    matrix[0, 2] = scale_x * (-0.5 * size_input[0] * math.cos(theta) +
                              0.5 * size_input[1] * math.sin(theta) +
                              0.5 * size_target[0])
    matrix[1, 0] = math.sin(theta) * scale_y
    matrix[1, 1] = math.cos(theta) * scale_y
    matrix[1, 2] = scale_y * (-0.5 * size_input[0] * math.sin(theta) -
                              0.5 * size_input[1] * math.cos(theta) +
                              0.5 * size_target[1])
    return matrix

def rotate_point(pt, angle_rad):
    """Rotate a point by an angle.

    Args:
        pt (list[float]): 2 dimensional point to be rotated
        angle_rad (float): rotation angle by radian

    Returns:
        list[float]: Rotated point.
    """
    assert len(pt) == 2
    sn, cs = np.sin(angle_rad), np.cos(angle_rad)
    new_x = pt[0] * cs - pt[1] * sn
    new_y = pt[0] * sn + pt[1] * cs
    rotated_pt = [new_x, new_y]

    return rotated_pt

def _get_3rd_point(a, b):
    """To calculate the affine matrix, three pairs of points are required. This
    function is used to get the 3rd point, given 2D points a & b.

    The 3rd point is defined by rotating vector `a - b` by 90 degrees
    anticlockwise, using b as the rotation center.

    Args:
        a (np.ndarray): point(x,y)
        b (np.ndarray): point(x,y)

    Returns:
        np.ndarray: The 3rd point.
    """
    assert len(a) == 2
    assert len(b) == 2
    direction = a - b
    third_pt = b + np.array([-direction[1], direction[0]], dtype=np.float32)

    return third_pt

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=(0., 0.),
                         inv=False):
    """Get the affine transform matrix, given the center/scale/rot/output_size.

    Args:
        center (np.ndarray[2, ]): Center of the bounding box (x, y).
        scale (np.ndarray[2, ]): Scale of the bounding box
            wrt [width, height].
        rot (float): Rotation angle (degree).
        output_size (np.ndarray[2, ] | list(2,)): Size of the
            destination heatmaps.
        shift (0-100%): Shift translation ratio wrt the width/height.
            Default (0., 0.).
        inv (bool): Option to inverse the affine transform direction.
            (inv=False: src->dst or inv=True: dst->src)

    Returns:
        np.ndarray: The transform matrix.
    """
    assert len(center) == 2
    assert len(scale) == 2
    assert len(output_size) == 2
    assert len(shift) == 2

    # pixel_std is 200.
    scale_tmp = scale * 200.0

    shift = np.array(shift)
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = rotate_point([0., src_w * -0.5], rot_rad)
    dst_dir = np.array([0., dst_w * -0.5])

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    src[2, :] = _get_3rd_point(src[0, :], src[1, :])

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
    dst[2, :] = _get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


class TopDownAffine:
    """Affine transform the image to make input.

    Required keys:'img', 'joints_3d', 'joints_3d_visible', 'ann_info','scale',
    'rotation' and 'center'.

    Modified keys:'img', 'joints_3d', and 'joints_3d_visible'.

    Args:
        use_udp (bool): To use unbiased data processing.
            Paper ref: Huang et al. The Devil is in the Details: Delving into
            Unbiased Data Processing for Human Pose Estimation (CVPR 2020).
    """

    def __init__(self, use_udp=False):
        self.use_udp = use_udp

    def __call__(self, results):
        image_size = results['image_size']

        img = results['img']
        c = results['center']
        s = results['scale']
        r = results['rotation']

        if self.use_udp:
            trans = get_warp_matrix(r, c * 2.0, image_size - 1.0, s * 200.0)
            if not isinstance(img, list):
                img = cv2.warpAffine(
                    img,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)
            else:
                img = [
                    cv2.warpAffine(
                        i,
                        trans, (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR) for i in img
                ]
        else:
            trans = get_affine_transform(c, s, r, image_size)
            if not isinstance(img, list):
                img = cv2.warpAffine(
                    img,
                    trans, (int(image_size[0]), int(image_size[1])),
                    flags=cv2.INTER_LINEAR)
            else:
                img = [
                    cv2.warpAffine(
                        i,
                        trans, (int(image_size[0]), int(image_size[1])),
                        flags=cv2.INTER_LINEAR) for i in img
                ]

        results['img'] = img
        return results
    
def _box2cs(model_input_size, box):
    """This encodes bbox(x,y,w,h) into (center, scale)

    Args:
        x, y, w, h

    Returns:
        tuple: A tuple containing center and scale.

        - np.ndarray[float32](2,): Center of the bbox (x, y).
        - np.ndarray[float32](2,): Scale of the bbox w & h.
    """

    x, y, w, h = box[:4]
    aspect_ratio = model_input_size[0] / model_input_size[1]
    center = np.array([x + w * 0.5, y + h * 0.5], dtype=np.float32)

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio

    # pixel std is 200.0
    scale = np.array([w / 200.0, h / 200.0], dtype=np.float32)
    scale = scale * 1.0 # original code, scale = scale * 1.25

    return center, scale

class ComposeTransforms:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, results):
        for transform in self.transforms:
            results = transform(results)
        return results

class BaseTransform:
    def transform(self, img):
        raise NotImplementedError

    def __call__(self, results):
        img = results['img']

        if isinstance(img, (list, tuple)):
            results['img'] = [self.transform(im) for im in img]
        else:
            results['img'] = self.transform(img)

        return results

class BGR2RGB(BaseTransform):
    def transform(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

class ToTensor(BaseTransform):
    def transform(self, img):
        return F.to_tensor(img)

class NormalizeTensor(BaseTransform):
    def __init__(self, 
                 mean=[0.485, 0.456, 0.406], 
                 std=[0.229, 0.224, 0.225]):
        self.mean = mean
        self.std = std

    def transform(self, img):
        return F.normalize(img, mean=self.mean, std=self.std)

model_config = {
    'ViTPose_base_simple_coco': {
        'model_cfg': 'ViTPose_base_simple_coco_256x192.py',
        'dataset_cfg': 'coco.py',
        'pth': 'vitpose-b-simple_half.pth',
        'url': 'https://github.com/developer0hye/onepose/releases/download/1.0.0/vitpose-b-simple_half.pth',
        'hash': '0408c829e344fe6f9d61eb16db5c863f'
    },
    'ViTPose_large_simple_coco': {
        'model_cfg': 'ViTPose_large_simple_coco_256x192.py',
        'dataset_cfg': 'coco.py',
        'pth': 'vitpose-l-simple_half.pth',
        'url': 'https://github.com/developer0hye/onepose/releases/download/1.0.0/vitpose-l-simple_half.pth',
        'hash': '6b35d98cdf0ac4838dbe9f4bb98dd38f'
    },
    'ViTPose_huge_simple_coco': {
        'model_cfg': 'ViTPose_huge_simple_coco_256x192.py',
        'dataset_cfg': 'coco.py',
        'pth': 'vitpose-h-simple_half.pth',
        'url': 'https://github.com/developer0hye/onepose/releases/download/1.0.0/vitpose-h-simple_half.pth',
        'hash': '319c1bf933f677bce2ad33da21304866'
    },
    'ViTPose_base_mpii': {
        'model_cfg': 'ViTPose_base_mpii_256x192.py',
        'dataset_cfg': 'mpii.py',
        'pth': 'vitpose-b-multi-mpii_half.pth',
        'url': 'https://github.com/developer0hye/onepose/releases/download/1.0.0/vitpose-b-multi-mpii_half.pth',
        'hash': '475eaab9c8fd78df77729cac7229c3e7'
    },
    'ViTPose_large_mpii': {
        'model_cfg': 'ViTPose_large_mpii_256x192.py',
        'dataset_cfg': 'mpii.py',
        'pth': 'vitpose-l-multi-mpii_half.pth',
        'url': 'https://github.com/developer0hye/onepose/releases/download/1.0.0/vitpose-l-multi-mpii_half.pth',
        'hash': '1c7a3a6d40e775b2ca376090bf8f55ed'
    },
    'ViTPose_huge_mpii': {
        'cfg': 'ViTPose_huge_mpii_256x192.py',
        'dataset_cfg': 'mpii.py',
        'pth': 'vitpose-h-multi-mpii_half.pth',
        'url': 'https://github.com/developer0hye/onepose/releases/download/1.0.0/vitpose-h-multi-mpii_half.pth',
        'hash': '38a0335fbc749c1bfb6b60f0b13e5c93'
    },
}

class Model(nn.Module):
    def __init__(self, 
                 model_name: str = 'ViTPose_huge_simple_coco') -> None:
        super().__init__()

        self.transforms = ComposeTransforms([
            BGR2RGB(),
            TopDownAffine(use_udp=True),
            ToTensor(),
            NormalizeTensor()
        ])

        file_path = pathlib.Path(os.path.abspath(__file__)).parent
        
        self.model_cfg = read_cfg(os.path.join(file_path, 'configs', model_config[model_name]['model_cfg']))
        self.model = vitpose.ViTPose(self.model_cfg.model)
        
        download_weights(model_name)
        
        ckpt = os.path.join(file_path, 'weights', model_config[model_name]['pth'])
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.eval()
        
        dataset_cfg = read_cfg(os.path.join(file_path.parent, 'datasets', model_config[model_name]['dataset_cfg']))
        self.keypoint_info = dataset_cfg.dataset_info['keypoint_info']

    @torch.no_grad()
    @torch.inference_mode()
    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            self.eval()
        
        device = next(self.parameters()).device
        
        img_height, img_width = x.shape[:2]
        center, scale = _box2cs(self.model_cfg.data_cfg['image_size'], [0, 0, img_width, img_height])      

        results = {'img': x,
                   'rotation': 0,
                   'center': center,
                   'scale': scale,
                   'image_size': np.array(self.model_cfg.data_cfg['image_size']),
                   }

        results = self.transforms(results)
        results['img'] = results['img'].to(device)
        
        out = self.model(results['img'][None, ...])
        out = out.cpu().numpy()
        
        out, maxvals = keypoints_from_heatmaps(out, 
                                      center=[center],
                                      scale=[scale], 
                                      unbiased=False, 
                                      post_process='default', 
                                      kernel=11, 
                                      valid_radius_factor=0.0546875, 
                                      use_udp=True, 
                                      target_type='GaussianHeatmap')
        out = out[0]
        maxvals = maxvals[0]
        out = {'points': out, 'confidence': maxvals}
        return out

def create_model(model_name: str = 'ViTPose_huge_simple_coco') -> Model:
    model = Model(model_name=model_name)
    return model

def list_models() -> List[str]:
    return list(model_config.keys())

def calc_file_hash(path):
    f = open(path, 'rb')
    data = f.read()
    hash = hashlib.md5(data).hexdigest()
    return hash

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_weights(model_name: str = 'ViTPose_huge_simple_coco') -> None:
    file_path = pathlib.Path(os.path.abspath(__file__)).parent
    os.makedirs(os.path.join(file_path, 'weights'), exist_ok=True)
    url = model_config[model_name]['url']
    ckpt = os.path.join(file_path, 'weights', model_config[model_name]['pth'])
    
    if os.path.exists(ckpt):
        if calc_file_hash(ckpt) == model_config[model_name]['hash']:
            print(f'Weights file for a {model_name} model is already downloaded.')
            return True
        os.remove(ckpt)

    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=ckpt, reporthook=t.update_to)

    assert model_config[model_name]['hash'] == calc_file_hash(ckpt), f'Weights file for a {model_name} model is corrupted.'
    
    
   