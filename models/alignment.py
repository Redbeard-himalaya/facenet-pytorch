import math
from typing import Union, Tuple
import numpy as np

import torch
from torch.nn import functional as F


def batch_rotate(x: torch.Tensor, degree: torch.Tensor) -> torch.Tensor:
    """
    https://discuss.pytorch.org/t/how-to-rotate-batch-of-images-by-a-batch-of-angles/187482
    Rotate batch of images [B, C, W, H] by a specific angles [B] 
    (counter clockwise)

    Parameters
    ----------
    x : torch.Tensor
      batch of images
    angle : torch.Tensor
      batch of angles

    Returns
    -------
    torch.Tensor
        batch of rotated images
    """
    if x.dim() != 4:
        raise TypeError(f"param x ({img.dim()}D) is not in 4D")
    angle = degree / 180 * torch.pi
    s = torch.sin(angle)
    c = torch.cos(angle)
    rot_mat = torch.stack((torch.stack([c, -s], dim=1),
                           torch.stack([s, c], dim=1)), dim=1)
    zeros = torch.zeros(rot_mat.size(0), 2).unsqueeze(2)
    aff_mat = torch.cat((rot_mat, zeros), 2)
    grid = F.affine_grid(aff_mat, x.size(), align_corners=True)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def alignment(img: torch.Tensor, landmark: np.ndarray) -> Tuple[torch.Tensor, float, int]:
    """
    Alignma given face with respect to the left and right eye coordinates.
    Left eye is the eye appearing on the left (right eye of the person). Left top point is (0, 0)
    Args:
        img (torch.Tensor): an image 3D or a batch of images 4D
        landmark (np.ndarray): a landmark of a face in 2D or a batch of landmarks of faces in 3D
    """

    if not isinstance(img, torch.Tensor):
        raise TypeError(f"param img ({type(img)}) is not torch.Tensor")
    if not isinstance(landmark, np.ndarray):
        raise TypeError(f"param landmark ({type(landmark)}) is not numpy.ndarray")
    img_dim = img.dim()
    lm_dim = len(landmark.shape)
    if img_dim < 3 or 4 < img_dim:
        raise TypeError(f"param img ({img_dim}D) is not 3D or 4D")
    if lm_dim < 2 or 3 < lm_dim:
        raise TypeError(f"param img ({img_dim}D) is not 3D or 4D")

    # change signle image and landmark to batch
    if img_dim == 3:
        img = img.unsqueeze(0)
    if lm_dim == 2:
        landmark = np.expand_dims(landmark, axis=0)

    # calculate angle in batch
    left_eye_x = torch.tensor(landmark[:,0,0].astype(np.float32)) # x batch
    left_eye_y = torch.tensor(landmark[:,0,1].astype(np.float32)) # y batch
    right_eye_x = torch.tensor(landmark[:,1,0].astype(np.float32)) # x batch
    right_eye_y = torch.tensor(landmark[:,1,1].astype(np.float32)) # y batch
    angle = torch.complex(right_eye_x - left_eye_x,
                          right_eye_y - left_eye_y).angle() * 180 / torch.pi
    # rotate in batch
    img = batch_rotate(img, angle)

    # -----------------------
    if img_dim == 3:
        return img[0]#, -angle[0]
    else:
        return img#, -angle
