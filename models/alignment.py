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


def find_euclidean_distance(
    source_representation: Union[np.ndarray, list], test_representation: Union[np.ndarray, list]
) -> float:
    """
    Find euclidean distance between 2 vectors
    Args:
        source_representation (numpy array or list)
        test_representation (numpy array or list)
    Returns
        distance
    """
    if isinstance(source_representation, list):
        source_representation = np.array(source_representation)

    if isinstance(test_representation, list):
        test_representation = np.array(test_representation)

    euclidean_distance = source_representation - test_representation
    euclidean_distance = np.sum(np.multiply(euclidean_distance, euclidean_distance))
    euclidean_distance = np.sqrt(euclidean_distance)
    return euclidean_distance


def alignment(img: torch.Tensor, landmark: np.ndarray) -> Tuple[torch.Tensor, float, int]:
    """
    Alignma given face with respect to the left and right eye coordinates.
    Left eye is the eye appearing on the left (right eye of the person). Left top point is (0, 0)
    Args:
        img (torch.Tensor): an image 3D or a batch of images 4D
        landmark (np.ndarray): landmark of the face
        left_eye (tuple): left eye coordinates.
            Left eye is appearing on the left of image (right eye of the person)
        right_eye (tuple): right eye coordinates.
            Right eye is appearing on the right of image (left eye of the person)
        nose (tuple): coordinates of nose
    """

    if not isinstance(img, torch.Tensor):
        raise TypeError(f"param img ({type(img)}) is not torch.Tensor")
    img_dim = img.dim()
    if img_dim < 3 or 4 < img_dim:
        raise TypeError(f"param img ({img_dim}D) is not 3D or 4D")

    left_eye = landmark[0]
    right_eye = landmark[1]
    nose = landmark[2]
    left_eye_x, left_eye_y = left_eye
    right_eye_x, right_eye_y = right_eye

    # -----------------------
    # find rotation direction
    if left_eye_y > right_eye_y:
        point_3rd = (right_eye_x, left_eye_y)
        direction = -1  # rotate same direction to clock
    else:
        point_3rd = (left_eye_x, right_eye_y)
        direction = 1  # rotate inverse direction of clock

    # -----------------------
    # find length of triangle edges

    a = find_euclidean_distance(np.array(left_eye), np.array(point_3rd))
    b = find_euclidean_distance(np.array(right_eye), np.array(point_3rd))
    c = find_euclidean_distance(np.array(right_eye), np.array(left_eye))

    # -----------------------
    # apply cosine rule
    if b != 0 and c != 0:  # this multiplication causes division by zero in cos_a calculation

        cos_a = (b * b + c * c - a * a) / (2 * b * c)

        # PR15: While mathematically cos_a must be within the closed range [-1.0, 1.0],
        # floating point errors would produce cases violating this
        # In fact, we did come across a case where cos_a took the value 1.0000000169176173
        # which lead to a NaN from the following np.arccos step
        cos_a = min(1.0, max(-1.0, cos_a))

        angle = np.arccos(cos_a)  # angle in radian
        angle = (angle * 180) / math.pi  # radian to degree

        # -----------------------
        # rotate base image

        if direction == -1:
            angle = 90 - angle

        # batch rotate
        if img_dim == 3:
            #img = torch.stack([img])
            img = img.unsqueeze(0)
        img = batch_rotate(img, torch.FloatTensor([direction * angle]))
    else:
        angle = 0.0  # Dummy value for undefined angle

    # -----------------------
    if img_dim == 3:
        return img[0], angle, direction
    else:
        return img, angle, direction


def rotate_facial_area(
    facial_area: Tuple[int, int, int, int], angle: float, direction: int, size: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    """
    Rotate the facial area around its center.

    Args:
        facial_area (tuple of int): Representing the (x1, y1, x2, y2) of the facial area.
        angle (float): Angle of rotation in degrees.
        direction (int): Direction of rotation (-1 for clockwise, 1 for counterclockwise).
        size (tuple of int): Tuple representing the size of the image (width, height).

    Returns:
        rotated_facial_area (tuple of int): Representing the new coordinates
            (x1, y1, x2, y2) of the rotated facial area.
    """
    # Angle in radians
    angle = angle * np.pi / 180

    height, weight = size

    # Translate the facial area to the center of the image
    x = (facial_area[0] + facial_area[2]) / 2 - weight / 2
    y = (facial_area[1] + facial_area[3]) / 2 - height / 2

    # Rotate the facial area
    x_new = x * np.cos(angle) + y * direction * np.sin(angle)
    y_new = -x * direction * np.sin(angle) + y * np.cos(angle)

    # Translate the facial area back to the original position
    x_new = x_new + weight / 2
    y_new = y_new + height / 2

    # Calculate projected coordinates after alignment
    x1 = x_new - (facial_area[2] - facial_area[0]) / 2
    y1 = y_new - (facial_area[3] - facial_area[1]) / 2
    x2 = x_new + (facial_area[2] - facial_area[0]) / 2
    y2 = y_new + (facial_area[3] - facial_area[1]) / 2

    # validate projected coordinates are in image's boundaries
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = min(int(x2), weight)
    y2 = min(int(y2), height)

    return (x1, y1, x2, y2)
