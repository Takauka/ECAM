import torch
import torch.nn as nn
from .normalizer import TrajNorm
import numpy as np
from sklearn.cluster import KMeans
from scipy.interpolate import BSpline

## shapes:
## S: scene size
## T: time length
## F: flattened trajectory (T*2)

class SingularSpace(nn.Module):
    r"""Singular space model

    Args:
        hyper_params (DotDict): The hyper-parameters
        norm_ori (bool): Whether to normalize the trajectory with the origin
        norm_rot (bool): Whether to normalize the trajectory with the rotation
        norm_sca (bool): Whether to normalize the trajectory with the scale"""

    def __init__(self, hyper_params, norm_ori=True, norm_rot=True, norm_sca=True):
        super().__init__()

        self.hyper_params = hyper_params
        self.t_obs, self.t_pred = hyper_params.obs_len, hyper_params.pred_len
        self.obs_svd, self.pred_svd = hyper_params.obs_svd, hyper_params.pred_svd
        self.k = hyper_params.k
        self.s = hyper_params.num_samples
        self.dim = hyper_params.traj_dim
        ## normalize the trajectory with the origin, rotation, and scale
        self.traj_normalizer = TrajNorm(ori=norm_ori, rot=norm_rot, sca=norm_sca)

        self.V_trunc_FK = nn.Parameter(torch.zeros((self.t_pred * self.dim, self.k)))
        ## Truncated singular space basis vectors: maps (T_obs*2) -> k
        self.V_obs_trunc_FK = nn.Parameter(torch.zeros((self.t_obs * self.dim, self.k)))
        ## Truncated singular space basis vectors: maps (T_pred*2) -> k
        self.V_pred_trunc_FK = nn.Parameter(torch.zeros((self.t_pred * self.dim, self.k)))

    def normalize_trajectory(self, obs_traj_BT2, pred_traj_BT2=None):
        r"""Trajectory normalization

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory (Optional, for training only)

        Returns:
            obs_traj_norm (torch.Tensor): The normalized observed trajectory
            pred_traj_norm (torch.Tensor): The normalized predicted trajectory
        """

        self.traj_normalizer.calculate_params(obs_traj_BT2)
        obs_traj_norm_BT2 = self.traj_normalizer.normalize(obs_traj_BT2)
        pred_traj_norm_BT2 = self.traj_normalizer.normalize(pred_traj_BT2) if pred_traj_BT2 is not None else None
        return obs_traj_norm_BT2, pred_traj_norm_BT2

    def denormalize_trajectory(self, traj_norm_NBT2):
        r"""Trajectory denormalization

        Args:
            traj_norm (torch.Tensor): The trajectory to be denormalized

        Returns:
            traj (torch.Tensor): The denormalized trajectory
        """

        traj_NBT2 = self.traj_normalizer.denormalize(traj_norm_NBT2)
        return traj_NBT2

    def to_Singular_space(self, traj_ST2, evec_FK):
        r"""Transform Euclidean trajectories to Singular space coordinates

        Args:
            traj (torch.Tensor): The trajectory to be transformed
            evec (torch.Tensor): The Singular space basis vectors

        Returns:
            C (torch.Tensor): The Singular space coordinates"""

        # Euclidean space -> Singular space
        ## Flattened time dimension (F)
        tdim = evec_FK.size(0)
        M_FS = traj_ST2.reshape(-1, tdim).T
        C_KS = evec_FK.T.detach() @ M_FS
        return C_KS

    def batch_to_Singular_space(self, traj_NBT2, evec_FK):
        # Euclidean space -> Singular space
        tdim = evec_FK.size(0)
        M_NFB = traj_NBT2.reshape(traj_NBT2.size(0), traj_NBT2.size(1), tdim).transpose(1, 2)
        C_NKB = evec_FK.T.detach() @ M_NFB
        return C_NKB

    def to_Euclidean_space(self, C_KS, evec_FK):
        r"""Transform Singular space coordinates to Euclidean trajectories

        Args:
            C (torch.Tensor): The Singular space coordinates
            evec (torch.Tensor): The Singular space basis vectors

        Returns:
            traj (torch.Tensor): The Euclidean trajectory"""

        ## Apparently not used

        # Singular space -> Euclidean
        ## Original time dimension (T = F / 2)
        t = evec_FK.size(0) // self.dim
        M_FS = evec_FK.detach() @ C_KS
        traj_ST2 = M_FS.T.reshape(-1, t, self.dim)
        return traj_ST2

    def batch_to_Euclidean_space(self, C_NKB, evec_FK):
        # Singular space -> Euclidean

        ## batch refers to the number of samples

        ## "batch size" = number of samples (N)
        b = C_NKB.size(0)
        ## Original time dimension (T = F / 2)
        t = evec_FK.size(0) // self.dim
        ## matrix evec_FK maps (F = 2T) -> K, and the C_NKB is a batch of
        ## N matrices of shape K x B, each of which represents the Singular space coordinates
        ## of the B pedestrians.
        M_NFB = evec_FK.detach() @ C_NKB
        traj_NBT2 = M_NFB.transpose(1, 2).reshape(b, -1, t, self.dim)
        return traj_NBT2

    def truncated_SVD(self, traj_BT2, k=None, full_matrices=False):
        r"""Truncated Singular Value Decomposition

        Args:
            traj (torch.Tensor): The trajectory to be decomposed
            k (int): The number of singular values and vectors to be computed
            full_matrices (bool): Whether to compute full-sized matrices

        Returns:
            U_trunc (torch.Tensor): The truncated left singular vectors
            S_trunc (torch.Tensor): The truncated singular values
            Vt_trunc (torch.Tensor): The truncated right singular vectors
        """

        ## Used in training only

        assert traj_BT2.size(2) == self.dim  # NTC
        k = self.k if k is None else k

        # Singular Value Decomposition
        M_FB = traj_BT2.reshape(-1, traj_BT2.size(1) * self.dim).T
        U, S, Vt = torch.linalg.svd(M_FB, full_matrices=full_matrices)

        # Truncated SVD
        U_trunc_FK, S_trunc_K, Vt_trunc_KB = U[:, :k], S[:k], Vt[:k, :]
        return U_trunc_FK, S_trunc_K, Vt_trunc_KB.T

    def parameter_initialization(self, obs_traj_BT2, pred_traj_BT2):
        r"""Initialize the Singular space basis vectors parameters (for training only)

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory

        Returns:
            pred_traj_norm (torch.Tensor): The normalized predicted trajectory
            V_pred_trunc (torch.Tensor): The truncated eigenvectors of the predicted trajectory

        Note:
            This function should be called once before training the model."""

        ## Training only

        # Normalize trajectory
        obs_traj_norm_BT2, pred_traj_norm_BT2 = self.normalize_trajectory(obs_traj_BT2, pred_traj_BT2)
        V_trunc_FK, _, _ = self.truncated_SVD(pred_traj_norm_BT2)

        # Pre-calculate the transformation matrix
        # Here, we use Irwin–Hall polynomial function

        ## degree of the polynomials
        degree=2
        ## likely two_t_*, since the trajectories are in 2d coords
        ## 2 x T_pred
        twot_win = self.dim * self.t_pred
        ## 2 x T_obs
        twot_hist=self.dim * self.t_obs
        ##
        steps = np.linspace(0., 1., twot_hist)
        knot = twot_win - degree + 1
        knots_qu = np.concatenate([np.zeros(degree), np.linspace(0, 1, knot), np.ones(degree)])
        C_hist_FF = np.zeros([twot_hist, twot_win])
        for i in range(twot_win):
            C_hist_FF[:, i] = BSpline(knots_qu, (np.arange(twot_win) == i).astype(float), degree, extrapolate=False)(steps)
        C_hist_FF = torch.from_numpy(C_hist_FF).float().to(obs_traj_BT2.device)

        ## make V_trunc work with the input history size (T_obs)
        ## by mapping T_obs*2 to T_win*2
        V_obs_trunc_FK = C_hist_FF @ V_trunc_FK
        ## since T_win = T_pred, we can just assign
        V_pred_trunc_FK = V_trunc_FK

        ## So, V_obs_trunc and V_pred_trunc are conceptually the same
        ## up to a projection applied to the input trajectory to make it
        ## of a standard size (T_win*2)

        # Register basis vectors as model parameters
        self.V_trunc_FK = nn.Parameter(V_trunc_FK.to(self.V_trunc_FK.device))
        self.V_obs_trunc_FK = nn.Parameter(V_obs_trunc_FK.to(self.V_obs_trunc_FK.device))
        self.V_pred_trunc_FK = nn.Parameter(V_pred_trunc_FK.to(self.V_pred_trunc_FK.device))

        # Reuse values for anchor generation
        return pred_traj_norm_BT2, V_pred_trunc_FK

    def projection(self, obs_traj, pred_traj=None):
        r"""Trajectory projection to the Singular space

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory (optional, for training only)

        Returns:
            C_obs (torch.Tensor): The observed trajectory in the Singular space
            C_pred (torch.Tensor): The predicted trajectory in the Singular space (optional, for training only)
        """

        # Trajectory Projection
        obs_traj_norm_ST2, pred_traj_norm_ST2 = self.normalize_trajectory(obs_traj, pred_traj)
        ## Singular space coordinates
        C_obs_KS = self.to_Singular_space(obs_traj_norm_ST2, evec_FK=self.V_obs_trunc_FK).detach()
        C_pred_KS = self.to_Singular_space(pred_traj_norm_ST2, evec_FK=self.V_pred_trunc_FK).detach() if pred_traj is not None else None
        return C_obs_KS, C_pred_KS

    def reconstruction(self, C_pred_KBN):
        r"""Trajectory reconstruction from the Singular space

        Args:
            C_pred (torch.Tensor): The predicted trajectory in the Singular space

        Returns:
            pred_traj (torch.Tensor): The predicted trajectory in the Euclidean space
        """

        C_pred_NKB = C_pred_KBN.permute(2, 0, 1)
        pred_traj_NBT2 = self.batch_to_Euclidean_space(C_pred_NKB, evec_FK=self.V_pred_trunc_FK)
        pred_traj_NBT2 = self.denormalize_trajectory(pred_traj_NBT2)

        return pred_traj_NBT2

    def forward(self, C_pred):
        r"""Alias for reconstruction"""

        return self.reconstruction(C_pred)
