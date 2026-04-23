import torch
import numpy as np 
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmdet.models.layers import inverse_sigmoid

def apply_ltrb(locations, pred_ltrb): 
        """
        :param locations:  (1, H, W, 2)
        :param pred_ltrb:  (N, H, W, 4) 
        """
        pred_boxes = torch.zeros_like(pred_ltrb)
        pred_boxes[..., 0] = (locations[..., 0] - pred_ltrb[..., 0])# x1
        pred_boxes[..., 1] = (locations[..., 1] - pred_ltrb[..., 1])# y1
        pred_boxes[..., 2] = (locations[..., 0] + pred_ltrb[..., 2])# x2
        pred_boxes[..., 3] = (locations[..., 1] + pred_ltrb[..., 3])# y2
        min_xy = pred_boxes[..., 0].new_tensor(0)
        max_xy = pred_boxes[..., 0].new_tensor(1)
        pred_boxes  = torch.where(pred_boxes < min_xy, min_xy, pred_boxes)
        pred_boxes  = torch.where(pred_boxes > max_xy, max_xy, pred_boxes)
        pred_boxes = bbox_xyxy_to_cxcywh(pred_boxes)


        return pred_boxes    

def apply_center_offset(locations, center_offset): 
        """
        :param locations:  (1, H, W, 2)
        :param pred_ltrb:  (N, H, W, 4) 
        """
        centers_2d = torch.zeros_like(center_offset)
        locations = inverse_sigmoid(locations)
        centers_2d[..., 0] = locations[..., 0] + center_offset[..., 0]  # x1
        centers_2d[..., 1] = locations[..., 1] + center_offset[..., 1]  # y1
        centers_2d = centers_2d.sigmoid()

        return centers_2d

def gaussian_2d(shape, sigma=1.0):
    """Generate gaussian map.

    Args:
        shape (list[int]): Shape of the map.
        sigma (float, optional): Sigma to generate gaussian map.
            Defaults to 1.

    Returns:
        np.ndarray: Generated gaussian map.
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_heatmap_gaussian(heatmap, center, radius, k=1):
    """Get gaussian masked heatmap.

    Args:
        heatmap (torch.Tensor): Heatmap to be masked.
        center (torch.Tensor): Center coord of the heatmap.
        radius (int): Radius of gaussian.
        K (int, optional): Multiple of masked_gaussian. Defaults to 1.

    Returns:
        torch.Tensor: Masked heatmap.
    """
    diameter = 2 * radius + 1
    gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = torch.from_numpy(
        gaussian[radius - top:radius + bottom,
                 radius - left:radius + right]).to(heatmap.device,
                                                   torch.float32)
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

