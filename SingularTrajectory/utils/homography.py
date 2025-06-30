import numpy as np
import torch


def project(xy: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
    """Projects points using the given homography matrix.

    Args:
        xy: Points to be projected.
            Shape: (*, 2).
        homography: Homography matrix.
            Shape: (3, 3).

    Returns:
        Projected points.
        Shape: (*, 2).
    """

    # Original shape.
    shape = list(xy.shape)
    shape[-1] = 1

    # Homogeneous coordinates.
    xy_hom = torch.cat((xy, torch.ones(shape, device=xy.device)), dim=-1)

    # Project the points.
    xy_proj = xy_hom @ homography.T

    # Euclidean coordinates.
    xy_proj = xy_proj / xy_proj[..., 2:]
    xy_proj = xy_proj[..., :2]

    return xy_proj

def project_batched(xy: torch.Tensor, homography: torch.Tensor) -> torch.Tensor:
    """Projects points using batched homography matrices.
    Args:
        xy: Points to be projected.
            Shape: (B, *, 2), where B is batch size
        homography: Batch of homography matrices.
            Shape: (B, 3, 3)
    Returns:
        Projected points.
        Shape: (B, *, 2)
    """

    # Reshape xy to (B, -1, 2) to flatten all middle dimensions.
    xy_reshaped = xy.view(xy.shape[0], -1, 2)

    # Homogeneous coordinates.
    ones_shape = xy_reshaped.shape[:-1] + (1,)
    ones = torch.ones(ones_shape, device=xy.device)
    xy_hom = torch.cat((xy_reshaped, ones), dim=-1)  # (B, -1, 3)

    # Project the points.
    xy_proj = torch.bmm(xy_hom, homography.transpose(1, 2))  # (B, -1, 3)

    # Euclidean coordinates
    xy_proj = xy_proj / xy_proj[..., 2:]
    xy_proj = xy_proj[..., :2]  # (B, -1, 2)

    # Restore original dimensions
    xy_proj = xy_proj.view(xy.shape)

    return xy_proj


def image2world(coord, H):
    r"""Convert image coordinates to world coordinates.

    Args:
        coord (np.ndarray or torch.tensor): Image coordinates, shape (..., 2).
        H (np.ndarray or torch.tensor): Homography matrix, shape (3, 3).

    Returns:
        np.ndarray: World coordinates.
    """

    assert coord.shape[-1] == 2
    assert H.shape == (3, 3)
    assert type(coord) == type(H)

    shape = coord.shape
    coord = coord.reshape(-1, 2)

    if isinstance(coord, np.ndarray):
        x, y = coord[..., 0], coord[..., 1]
        world = (H @ np.stack([x, y, np.ones_like(x)], axis=-1).T).T
        world = world / world[..., [2]]
        world = world[..., :2]

    elif isinstance(coord, torch.Tensor):
        x, y = coord[..., 0], coord[..., 1]
        world = (H @ torch.stack([x, y, torch.ones_like(x)], dim=-1).T).T
        world = world / world[..., [2]]
        world = world[..., :2]

    else:
        raise NotImplementedError

    return world.reshape(shape)


def world2image(coord, H, transpose=False):
    r"""Convert world coordinates to image coordinates.

    Args:
        coord (np.ndarray or torch.tensor): World coordinates, shape (..., 2).
        H (np.ndarray or torch.tensor): Homography matrix, shape (3, 3).

    Returns:
        np.ndarray: Image coordinates.
    """

    assert coord.shape[-1] == 2
    assert H.shape == (3, 3)
    assert type(coord) == type(H)

    shape = coord.shape
    coord = coord.reshape(-1, 2)

    if isinstance(coord, np.ndarray):
        x, y = coord[..., 0], coord[..., 1]
        image = (np.linalg.inv(H) @ np.stack([x, y, np.ones_like(x)], axis=-1).T).T
        image = image / image[..., [2]]
        image = image[..., :2]

    elif isinstance(coord, torch.Tensor):
        x, y = coord[..., 0], coord[..., 1]
        image = (torch.linalg.inv(H) @ torch.stack([x, y, torch.ones_like(x)], dim=-1).T).T
        image = image / image[..., [2]]
        image = image[..., :2]

    else:
        raise NotImplementedError

    return image.reshape(shape)


def generate_homography(shift_w: float=0, shift_h: float=0, rotate: float=0, scale: float=1):
    r"""Generate a homography matrix.

    Args:
        shift (float): Shift in x and y direction.
        rotate (float): Rotation angle in radian.
        scale (float): Scale factor.

    Returns:
        np.ndarray: Homography matrix, shape (3, 3).
    """

    H = np.eye(3)
    H[0, 2] = shift_w
    H[1, 2] = shift_h
    H[2, 2] = scale

    if rotate != 0:
        # rotation matrix
        R = np.array([[np.cos(rotate), -np.sin(rotate), 0],
                      [np.sin(rotate), np.cos(rotate), 0],
                      [0, 0, 1]])
        H = H @ R

    return H
