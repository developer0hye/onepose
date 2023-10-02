from torchvision.transforms import functional as F
import math
import numpy as np
import cv2

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
