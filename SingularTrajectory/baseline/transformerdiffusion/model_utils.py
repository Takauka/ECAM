import os
import subprocess
from typing import Literal

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import cv2 as cv
import yaml

import utils


def gen_noise(shape: tuple[int, ...],
              noise_distrib: Literal['gaussian', 'uniform'],
              generator: torch.Generator | None = None,
              device=None):
    if noise_distrib == 'gaussian':
        return torch.randn(shape, generator=generator, device=device)
    elif noise_distrib == 'uniform':
        return torch.rand(shape, generator=generator, device=device)
    else:
        raise ValueError(f'Noise type {noise_distrib} not supported')


def handle_noise(num_samples: int,
                 scene_size: int,
                 noise_dim: int,
                 noise_distrib: Literal['gaussian', 'uniform'],
                 noise_type: str,
                 noise: torch.Tensor | None,
                 device: torch.device) -> torch.Tensor:
    sampling_required = num_samples > 1 or noise is not None
    if sampling_required and noise_dim <= 0:
        raise ValueError('Cannot sample multiple trajectories '
                         'without noise')

    if noise is not None:
        # Check if noise is of correct shape.
        if noise_type == 'local':
            if noise.shape != (num_samples, scene_size, noise_dim):
                raise ValueError(
                    f'Noise shape must be '
                    f'({num_samples}, {scene_size}, {noise_dim})'
                )
            noise_KSL = noise

        else:
            if noise.shape != (num_samples, noise_dim):
                raise ValueError(
                    f'Noise shape must be ({num_samples}, {noise_dim})'
                )
            noise_KSL = noise.unsqueeze(1).repeat(1, scene_size, 1)

    else:
        # Generate noise.
        if noise_type == 'local':
            noise_KSL = gen_noise(
                (num_samples, scene_size, noise_dim),
                noise_distrib,
                device=device
            )
        else:
            noise_KL = gen_noise(
                (num_samples, noise_dim),
                noise_distrib,
                device=device
            )
            noise_KSL = noise_KL.unsqueeze(1).repeat(1, scene_size, 1)

    return noise_KSL


@torch.no_grad()
def check_env_collisions(traj_BP2,
                        map_mask_1HW,
                        scene_transform_matrix,
                        homography_meters2mask):
    """Checks if the given trajectories go over the non-walkable area.

    Args:
        traj_BP2: Trajectories in meters. Shape: (batch_size, pred_len, 2).
        map_mask_1HW: Map mask tensor. Shape: (1, height, width).
        scene_transform_matrix: Scene transform matrix.
        homography_meters2mask: Homography matrix to convert meters to mask coordinates.

    Returns:
        Boolean tensor indicating collisions. Shape: (batch_size,).
    """
    # Early return if no map mask
    if map_mask_1HW is None:
        return torch.zeros(traj_BP2.shape[0], dtype=torch.bool, device=traj_BP2.device)

    # Get dimensions
    num_traj, _, _ = traj_BP2.shape
    _, H, W = map_mask_1HW.shape

    # Convert map mask to boolean once
    map_mask_bool_1HW = map_mask_1HW > 0.5

    # Combine transformations into a single matrix multiplication
    combined_transform = torch.matmul(
        torch.inverse(scene_transform_matrix),
        homography_meters2mask
    )

    # Transform all trajectories at once
    traj_BP3 = torch.cat((traj_BP2, torch.ones_like(traj_BP2[..., :1])), dim=-1)
    traj_BP2 = torch.matmul(traj_BP3, combined_transform.T)[..., :2]

    # Calculate bounds for all trajectories at once
    x_coords = traj_BP2[..., 0]
    y_coords = traj_BP2[..., 1]

    # Check bounds once for all points
    in_bounds = (y_coords >= 0) & (y_coords < H) & (x_coords >= 0) & (x_coords < W)

    # Vectorized collision detection
    collisions = torch.zeros(num_traj, dtype=torch.bool, device=traj_BP2.device)

    for i in range(num_traj):
        if not in_bounds[i].any():
            continue

        # Extract relevant coordinates for this trajectory
        valid_y = y_coords[i][in_bounds[i]].long()
        valid_x = x_coords[i][in_bounds[i]].long()

        # Check if any point in trajectory overlaps with non-walkable area
        if not map_mask_bool_1HW[0, valid_y, valid_x].all():
            collisions[i] = True

    return collisions

@torch.no_grad()
def check_env_collisions_precise(traj_BP2,
                        map_mask_1HW,
                        scene_transform_matrix,
                        homography_meters2mask):
    """Checks if the given trajectories go over the non-walkable area.

    Args:
        traj_BP2: Trajectories in meters. Shape: (batch_size, pred_len, 2).
        map_mask_1HW: Map mask tensor. Shape: (1, height, width).
        scene_transform_matrix: Scene transform matrix.
        homography_meters2mask: Homography matrix to convert meters to mask coordinates.

    Returns:
        Boolean tensor indicating collisions. Shape: (batch_size,).
    """
    # Early return if no map mask
    if map_mask_1HW is None:
        return torch.zeros(traj_BP2.shape[0], dtype=torch.bool, device=traj_BP2.device)

    # Get dimensions
    num_traj, pred_len, _ = traj_BP2.shape
    _, H, W = map_mask_1HW.shape

    # Convert map mask to boolean once
    map_mask_bool_1HW = map_mask_1HW > 0.5

    # Combine transformations into a single matrix multiplication
    combined_transform = torch.matmul(
        torch.inverse(scene_transform_matrix),
        homography_meters2mask
    )

    # Transform all trajectories at once
    traj_BP3 = torch.cat((traj_BP2, torch.ones_like(traj_BP2[..., :1])), dim=-1)
    traj_BP2 = torch.matmul(traj_BP3, combined_transform.T)[..., :2]

    # Calculate bounds for all trajectories at once
    x_coords_BP = traj_BP2[..., 0]
    y_coords_BP = traj_BP2[..., 1]

    # Check bounds once for all points
    in_bounds_BP = (y_coords_BP >= 0) & (y_coords_BP < H) & (x_coords_BP >= 0) & (x_coords_BP < W)

    # Vectorized collision detection
    collisions_BP = torch.zeros((num_traj, pred_len), dtype=torch.bool, device=traj_BP2.device)

    for i in range(num_traj):
        if not in_bounds_BP[i].any():
            continue

        # Extract relevant coordinates for this trajectory
        valid_y_P = y_coords_BP[i][in_bounds_BP[i]].long()
        valid_x_P = x_coords_BP[i][in_bounds_BP[i]].long()

        # Check if any point in trajectory overlaps with non-walkable area
        collisions_BP[i, in_bounds_BP[i]] = ~map_mask_bool_1HW[0, valid_y_P, valid_x_P]

    return collisions_BP


def augment_traj_resolution(traj_ST2: torch.Tensor, parts: int) -> torch.Tensor:
    """Augment trajectory resolution by adding `parts` interpolated points
    between each pair of consecutive points."""

    # S: scene size (number of trajectories)
    # T: trajectory length
    # P: number of parts (or number of parts - 1)
    # 2: x, y coordinates

    S, _, _ = traj_ST2.shape

    # Create interpolation coefficients
    coeffs_P = torch.linspace(0, 1, parts + 2, device=traj_ST2.device)
    coeffs_P = coeffs_P[:-1]
    coeffs_P1 = coeffs_P[:, None]

    # Start and end points for the interpolation, with an extra dimension
    # for the parts.
    start_points_ST12 = traj_ST2[:, :-1, None, :]
    end_points_ST12 = traj_ST2[:, 1:, None, :]

    # Interpolate between the start and end points.
    interpolated_points_STL2 = \
        start_points_ST12 + coeffs_P1 * (end_points_ST12 - start_points_ST12)

    # Concatenate the interpolated points along the parts dimension
    # into the time dimension.
    interpolated_points_ST2 = interpolated_points_STL2.reshape(S, -1, 2)

    # Add the last point of the original trajectory.
    interpolated_points_ST2 = \
        torch.cat((interpolated_points_ST2, traj_ST2[:, -1:, :]), dim=1)

    return interpolated_points_ST2


@torch.no_grad()
def extract_patches(scene_SO2: torch.Tensor,
                    map_mask_1HW: torch.Tensor,
                    scene_transform_matrix: torch.Tensor,
                    homography_meters2mask: torch.Tensor) -> torch.Tensor:
    """Extract patches from the map mask for each person in the scene.

    Args:
        scene_SO2: Scene tensor. Expected in absolute coordinates and
            in meters. Need at least 2 timesteps.
            Shape: (scene_size, obs_len, 2).
        map_mask_1HW: Map mask tensor. Shape: (1, height, width).
        scene_transform_matrix: Scene transform matrix.
            The code will use the inverse of this matrix to undo
            data augmentation, for aligning the scene with the map.
        dataset_name: Name of the dataset.
        homography_meters2mask: Homography matrix to convert meters to mask
            pixel coordinates.

    Returns:
        Mask patches tensor. Shape: (scene_size, 1, patch_size, patch_size).
    """

    PATCH_SIZE_M = 10      # meters
    PATCH_SIZE_PX = 100    # pixels
    BACK_DIST_M = 1        # meters
    BACK_DIST_PX = 10      # pixels

    # Compute inverse matrix to undo transform (data augmentation).
    inv_transform_matrix = torch.inverse(scene_transform_matrix)

    # Last 2 positions.
    curr_pos_S2 = scene_SO2[:, -1, :]
    prev_pos_S2 = scene_SO2[:, -2, :]

    # To homogeneous coordinates.
    ones = torch.ones(curr_pos_S2.shape[0], 1, device=curr_pos_S2.device)
    curr_pos_S3 = torch.cat((curr_pos_S2, ones), dim=1)
    prev_pos_S3 = torch.cat((prev_pos_S2, ones), dim=1)

    # Apply inverse transform (undo data augmentation).
    curr_pos_S3 = torch.matmul(inv_transform_matrix, curr_pos_S3.T).T
    prev_pos_S3 = torch.matmul(inv_transform_matrix, prev_pos_S3.T).T

    # Back to world coordinates.
    curr_pos_S2 = curr_pos_S3[:, :2]
    prev_pos_S2 = prev_pos_S3[:, :2]

    # Convert world coordinates to mask pixel coordinates.
    curr_pos_S2 = utils.project(curr_pos_S2, homography_meters2mask)
    prev_pos_S2 = utils.project(prev_pos_S2, homography_meters2mask)

    # equals_S = np.isclose(curr_pos_S2, prev_pos_S2).all(axis=1)
    equals_S = (curr_pos_S2 == prev_pos_S2).all(axis=1)

    if equals_S.any():
        # If positions are the same, randomly perturb the prev.
        prev_pos_S2 = prev_pos_S2.clone()

        # Sample angles uniformly between 0 and 2pi for the perturbation.
        angles_S = 2 * torch.pi * torch.rand(equals_S.sum(),
                                             device=scene_SO2.device)
        delta_S2 = torch.stack([torch.cos(angles_S),
                                torch.sin(angles_S)], dim=-1)

        # Apply perturbations to the previous positions.
        prev_pos_S2[equals_S] += delta_S2

    # Compute forward direction.
    fwd_dir_S2 = curr_pos_S2 - prev_pos_S2
    fwd_dir_S2 /= torch.norm(fwd_dir_S2, dim=1, keepdim=True)

    # Compute left direction.
    left_dir_S2 = torch.stack((-fwd_dir_S2[:, 1], fwd_dir_S2[:, 0]), dim=1)

    # Compute back position.
    back_pos_S2 = curr_pos_S2 - BACK_DIST_PX * fwd_dir_S2

    # Compute corners.
    back_left_S2 = back_pos_S2 + left_dir_S2 * PATCH_SIZE_PX / 2
    back_right_S2 = back_pos_S2 - left_dir_S2 * PATCH_SIZE_PX / 2
    front_left_S2 = back_left_S2 + fwd_dir_S2 * PATCH_SIZE_PX
    front_right_S2 = back_right_S2 + fwd_dir_S2 * PATCH_SIZE_PX

    # Convert to numpy since opencv requires it.
    corners_4S2 = np.array([back_left_S2.cpu(),
                            back_right_S2.cpu(),
                            front_left_S2.cpu(),
                            front_right_S2.cpu()],
                            dtype=np.float32)
    target_corners_42 = np.array([[0, 0],
                                    [PATCH_SIZE_PX, 0],
                                    [0, PATCH_SIZE_PX],
                                    [PATCH_SIZE_PX, PATCH_SIZE_PX]],
                                    dtype=np.float32)

    # cv.warpAffine requires (H, W) format.
    map_mask_HW = map_mask_1HW.squeeze(0).cpu().numpy()

    mask_patches = []
    for i in range(scene_SO2.shape[0]):
        # Extract patches.
        # Affine transform computation requires 3 points.
        patch_affine_transform = cv.getAffineTransform(
            np.ascontiguousarray(corners_4S2[:3, i]),
            target_corners_42[:3]
        )

        mask_patch = cv.warpAffine(map_mask_HW,
                                    patch_affine_transform,
                                    (PATCH_SIZE_PX, PATCH_SIZE_PX),
                                    borderValue=255)
        mask_patch = mask_patch / 255.0
        mask_patches.append(mask_patch)

    mask_patches = np.array(mask_patches)

    mask_tensor_S1HW = \
        torch.from_numpy(mask_patches).\
                unsqueeze(1).\
                float().\
                to(scene_SO2.device)

    return mask_tensor_S1HW, curr_pos_S2, prev_pos_S2



@torch.no_grad()
def extract_patches_batched(traj_BO2: torch.Tensor,
                            map_mask_B1HW: torch.Tensor,
                            scene_transform_matrix_B33: torch.Tensor,
                            homography_meters2mask_B33: torch.Tensor,
                            patch_size_px: int,
                            back_dist_px: int) -> torch.Tensor:
    """
    Batched patch extraction using torch functions (replacing OpenCV).
    See original function for argument descriptions.
    Assumes map_mask_B1HW is a tensor of shape (B,1,H,W) with mask values in [0,255].
    """
    device = traj_BO2.device
    B = traj_BO2.shape[0]

    # 1. Compute current and previous positions in world coordinates.
    curr_pos_B2 = traj_BO2[:, -1, :]  # (B,2)
    prev_pos_B2 = traj_BO2[:, -2, :]  # (B,2)

    # Combine the scene transform and the meters-to-mask homography.
    combined_transform_B33 = torch.bmm(
        scene_transform_matrix_B33, homography_meters2mask_B33
    )

    # Project world coordinates into mask pixel coordinates.
    curr_pos_B2 = utils.project_batched(curr_pos_B2, combined_transform_B33)
    prev_pos_B2 = utils.project_batched(prev_pos_B2, combined_transform_B33)

    # If current and previous positions coincide, perturb the previous position.
    equals_B = (curr_pos_B2 == prev_pos_B2).all(dim=1)
    if equals_B.any():
        prev_pos_B2 = prev_pos_B2.clone()
        angles = 2 * torch.pi * torch.rand(equals_B.sum(), device=device)
        delta = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
        prev_pos_B2[equals_B] += delta

    # 2. Compute the (unit) forward and left directions.
    fwd_dir_B2 = curr_pos_B2 - prev_pos_B2
    fwd_dir_B2 = fwd_dir_B2 / torch.norm(fwd_dir_B2, dim=1, keepdim=True)
    left_dir_B2 = torch.stack([-fwd_dir_B2[:, 1], fwd_dir_B2[:, 0]], dim=1)

    # 3. Compute the “back” position and the three corners we need.
    #    We define the patch such that its “back” is at a fixed offset behind the person.
    back_pos_B2 = curr_pos_B2 - back_dist_px * fwd_dir_B2  # (B,2)
    # The three corners (in the map mask image) that we will use to define an affine transform:
    #    - back_left: the back-center shifted left half the patch width.
    #    - back_right: the back-center shifted right half the patch width.
    #    - front_left: the front-left (obtained by moving forward from back_left by patch_size_px).
    back_left_B2  = back_pos_B2 + left_dir_B2 * (patch_size_px / 2)
    back_right_B2 = back_pos_B2 - left_dir_B2 * (patch_size_px / 2)
    front_left_B2 = back_left_B2 + fwd_dir_B2 * patch_size_px
    # (We could also compute front_right, but three points suffice.)

    # 4. For each patch, compute the affine transform that maps patch pixel coordinates (in the target)
    #    to the corresponding pixel coordinates in the map.
    #
    # In OpenCV, one would use:
    #   patch_affine_transform = cv.getAffineTransform(src_points, dst_points)
    # where src_points are the 3 computed corners and dst_points are the 3 corners of a square:
    #   target_corners = [[0, 0], [patch_size_px, 0], [0, patch_size_px]].
    #
    # Here we compute it directly.
    # For each patch, let:
    #   src0 = back_left, src1 = back_right, src2 = front_left.
    # We want M (2x3) such that:
    #   M @ [0, 0, 1]^T = src0,
    #   M @ [patch_size_px, 0, 1]^T = src1,
    #   M @ [0, patch_size_px, 1]^T = src2.
    #
    # This gives:
    #   M[:,2] = src0,
    #   M[:,0] = (src1 - src0) / patch_size_px,
    #   M[:,1] = (src2 - src0) / patch_size_px.
    #
    M = torch.empty(B, 2, 3, device=device)
    M[:, :, 2] = back_left_B2  # broadcast: M[i, :, 2] = back_left of patch i.
    M[:, :, 0] = (back_right_B2 - back_left_B2) / patch_size_px
    M[:, :, 1] = (front_left_B2 - back_left_B2) / patch_size_px

    # 5. Convert this transform into the “normalized” coordinates required by grid_sample.
    #
    # grid_sample expects an affine matrix theta of shape (B,2,3) such that for an output grid (in normalized
    # coordinates, i.e. in [-1,1]), the source sampling point is computed as:
    #    [x_s, y_s] = theta @ [x_t, y_t, 1]
    #
    # Our computed matrix M maps patch pixel coordinates (in [0, patch_size_px]) to map pixel coordinates.
    # To use grid_sample we must:
    #
    #   1. Convert patch pixel coordinates (target) to normalized coordinates.
    #   2. Convert map pixel coordinates (source) to normalized coordinates.
    #
    # That is, we need to incorporate two normalization transforms.
    #
    # Define T_patch_inv: 3x3 transform converting patch normalized coordinates (in [-1,1]) into patch pixels.
    # With the convention that:
    #   x_pixel = ( (x_norm + 1) * (patch_size_px - 1) / 2 )
    #
    # Similarly, define T_in: 3x3 transform converting map pixel coordinates into normalized coordinates.
    # For an input map of size (H, W) (width = W, height = H):
    #   x_norm = 2 * x_pixel/(W - 1) - 1
    #   y_norm = 2 * y_pixel/(H - 1) - 1
    #
    # Thus we set:
    H = map_mask_B1HW.shape[2]
    W = map_mask_B1HW.shape[3]

    # T_in: from input pixels to normalized coordinates.
    T_in = torch.tensor([[2/(W-1), 0, -1],
                           [0, 2/(H-1), -1],
                           [0, 0, 1]], device=device, dtype=torch.float32)
    # T_patch_inv: from normalized patch coordinates to patch pixels.
    T_patch_inv = torch.tensor([[(patch_size_px - 1) / 2, 0, (patch_size_px - 1) / 2],
                                 [0, (patch_size_px - 1) / 2, (patch_size_px - 1) / 2],
                                 [0, 0, 1]], device=device, dtype=torch.float32)
    # Expand these to batch (they are the same for all patches).
    T_in = T_in.unsqueeze(0).expand(B, 3, 3)       # (B,3,3)
    T_patch_inv = T_patch_inv.unsqueeze(0).expand(B, 3, 3)  # (B,3,3)

    # Augment M (which is 2x3) to a 3x3 by adding [0, 0, 1] as the last row.
    last_row = torch.tensor([0, 0, 1], device=device, dtype=torch.float32).view(1, 1, 3).expand(B, 1, 3)
    M_aug = torch.cat([M, last_row], dim=1)  # (B,3,3)

    # Our overall mapping from patch normalized coordinates to input normalized coordinates is:
    #   theta_full = T_in @ M_aug @ T_patch_inv
    theta_full = torch.bmm(torch.bmm(T_in, M_aug), T_patch_inv)  # (B, 3, 3)
    # grid_sample only requires the 2x3 part.
    theta = theta_full[:, :2, :]  # (B,2,3)

    # 6. Create the grid and sample.
    # Normalize: grid_sample expects the output grid in normalized coordinates.
    grid = F.affine_grid(theta, size=(B, 1, patch_size_px, patch_size_px),
                         align_corners=True)

    # Note: The OpenCV version uses borderValue=255, then divides the result by 255.
    # Here we assume that map_mask_B1HW has values in [0, 255]. We first convert to float and scale.
    map_mask_norm = map_mask_B1HW.float() / 255.0

    # Use grid_sample. We use mode='bilinear' (as in warpAffine) and padding_mode='border'
    # so that out-of-bound locations are filled with the border value (which, if the border is 255,
    # becomes 1 after normalization).
    mask_patches = F.grid_sample(map_mask_norm, grid, mode='bilinear',
                                 padding_mode='border', align_corners=True)
    # mask_patches is (B, 1, patch_size_px, patch_size_px)

    return mask_patches, curr_pos_B2, prev_pos_B2


def git_info():
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()

    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass

    return sha, diff, branch
