import torch


class TrajNorm:
    r"""Normalize trajectory with shape (num_peds, length_of_time, 2)

    Args:
        ori (bool): Whether to normalize the trajectory with the origin
        rot (bool): Whether to normalize the trajectory with the rotation
        sca (bool): Whether to normalize the trajectory with the scale
    """

    def __init__(self, ori=True, rot=True, sca=True):
        self.ori, self.rot, self.sca = ori, rot, sca
        self.traj_ori_B12, self.traj_rot_B22, self.traj_sca_B11 = None, None, None

    def calculate_params(self, traj_BT2):
        r"""Calculate the normalization parameters"""

        if self.ori:
            ## origin is the last point of the trajectory
            self.traj_ori_B12 = traj_BT2[:, [-1]]
        if self.rot:
            ## person direction: use last 3 points to calculate the direction,
            ## if the trajectory length is less than 3, use the last 2 points
            if traj_BT2.size(1) <= 2:
                ## last - prev
                dir_B2 = traj_BT2[:, -1] - traj_BT2[:, -2]
            else:
                ## last - prev_prev
                dir_B2 = traj_BT2[:, -1] - traj_BT2[:, -3]
            ## rotation matrix
            rot_B = torch.atan2(dir_B2[:, 1], dir_B2[:, 0])
            self.traj_rot_B22 = torch.stack([torch.stack([rot_B.cos(), -rot_B.sin()], dim=1),
                                         torch.stack([rot_B.sin(), rot_B.cos()], dim=1)], dim=1)
        if self.sca:
            ## scale 1/norm(last - prev)
            ## (use last 3 points if possible, otherwise use last 2 points,
            ## (double the scale if using last 3 points)
            ## normalization makes the last speed to be close to 1
            if traj_BT2.size(1) <= 2:
                self.traj_sca_B11 = 1. / (traj_BT2[:, -1] - traj_BT2[:, -2]).norm(p=2, dim=-1)[:, None, None] * 1
            else:
                self.traj_sca_B11 = 1. / (traj_BT2[:, -1] - traj_BT2[:, -3]).norm(p=2, dim=-1)[:, None, None] * 2
            # self.traj_sca[self.traj_sca.isnan() | self.traj_sca.isinf()] = 1e2

    def get_params(self):
        r"""Get the normalization parameters"""

        return self.ori, self.rot, self.sca, self.traj_ori_B12, self.traj_rot_B22, self.traj_sca_B11

    def set_params(self, ori, rot, sca, traj_ori, traj_rot, traj_sca):
        r"""Set the normalization parameters"""

        self.ori, self.rot, self.sca = ori, rot, sca
        self.traj_ori_B12, self.traj_rot_B22, self.traj_sca_B11 = traj_ori, traj_rot, traj_sca

    def normalize(self, traj_BT2):
        r"""Normalize the trajectory"""

        ## likely works even with NBT2

        if self.ori:
            traj_BT2 = traj_BT2 - self.traj_ori_B12
        if self.rot:
            traj_BT2 = traj_BT2 @ self.traj_rot_B22
        if self.sca:
            traj_BT2 = traj_BT2 * self.traj_sca_B11
        return traj_BT2

    def denormalize(self, traj_NBT2, batch_size=16384):
        """Denormalize the trajectory in batches to avoid memory issues.

        Args:
            traj_NBT2: Input trajectory tensor of shape (N, B, T, 2)
            batch_size: Number of batches to process at once

        Returns:
            Denormalized trajectory tensor of shape (N, B, T, 2)
        """

        _, B, _, _ = traj_NBT2.shape

        # Handle empty or single-batch cases
        if B == 0:
            return traj_NBT2
        if B <= batch_size:
            # If the batch is small enough, process it all at once
            if self.sca:
                traj_NBT2 = traj_NBT2 / self.traj_sca_B11
            if self.rot:
                traj_NBT2 = traj_NBT2 @ self.traj_rot_B22.transpose(-1, -2)
            if self.ori:
                traj_NBT2 = traj_NBT2 + self.traj_ori_B12
            return traj_NBT2

        result = []

        # Process the trajectory in batches along the B dimension
        for start_idx in range(0, B, batch_size):
            end_idx = min(start_idx + batch_size, B)
            # Select batch along dimension 1
            batch = traj_NBT2[:, start_idx:end_idx]

            # Apply transformations to the current batch
            if self.sca:
                # Index sca along the correct batch dimension
                batch = batch / self.traj_sca_B11[start_idx:end_idx]
            if self.rot:
                # Index rot along the correct batch dimension
                batch = batch @ self.traj_rot_B22[start_idx:end_idx].transpose(-1, -2)
            if self.ori:
                # Index ori along the correct batch dimension
                batch = batch + self.traj_ori_B12[start_idx:end_idx]

            result.append(batch)

        # Concatenate all batches back together along dimension 1
        return torch.cat(result, dim=1)
