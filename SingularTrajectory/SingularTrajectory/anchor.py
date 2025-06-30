import torch
import torch.nn as nn
from .kmeans import BatchKMeans
from sklearn.cluster import KMeans
import numpy as np
from .homography import image2world, world2image


class AdaptiveAnchor(nn.Module):
    r"""Adaptive anchor model

    Args:
        hyper_params (DotDict): The hyper-parameters
    """

    def __init__(self, hyper_params):
        super().__init__()

        self.hyper_params = hyper_params
        ## embedding size
        self.k = hyper_params.k
        ## number of samples
        self.s = hyper_params.num_samples
        ## trajectory dimension (2 for 2D data)
        self.dim = hyper_params.traj_dim

        self.C_anchor_KN = nn.Parameter(torch.zeros((self.k, self.s)))

    def to_Singular_space(self, traj_BT2, evec_FK):
        r"""Transform Euclidean trajectories to Singular space coordinates

        Args:
            traj (torch.Tensor): The trajectory to be transformed
            evec (torch.Tensor): The Singular space basis vectors

        Returns:
            C (torch.Tensor): The Singular space coordinates"""

        # Euclidean space -> Singular space
        tdim = evec_FK.size(0)
        M_FB = traj_BT2.reshape(-1, tdim).T
        C_KB = evec_FK.T.detach() @ M_FB
        return C_KB

    def batch_to_Singular_space(self, traj_NBT2, evec_FK):
        ## not used and likely wrong (transposition operation)
        ## the correct method is in space file

        # Euclidean space -> Singular space
        tdim = evec_FK.size(0)
        M = traj_NBT2.reshape(-1, tdim).transpose(1, 2)
        C = evec_FK.T.detach() @ M
        return C

    def to_Euclidean_space(self, C, evec):
        r"""Transform Singular space coordinates to Euclidean trajectories

        Args:
            C (torch.Tensor): The Singular space coordinates
            evec (torch.Tensor): The Singular space basis vectors

        Returns:
            traj (torch.Tensor): The Euclidean trajectory"""

        # Singular space -> Euclidean
        t = evec.size(0) // self.dim
        M = evec.detach() @ C
        traj = M.T.reshape(-1, t, self.dim)
        return traj

    def batch_to_Euclidean_space(self, C, evec):
        # Singular space -> Euclidean
        b = C.size(0)
        t = evec.size(0) // self.dim
        M = evec.detach() @ C
        traj = M.transpose(1, 2).reshape(b, -1, t, self.dim)
        return traj

    def anchor_initialization(self, pred_traj_norm_BT2, V_pred_trunc_FK):
        r"""Anchor initialization on Singular space

        Args:
            pred_traj_norm (torch.Tensor): The normalized predicted trajectory
            V_pred_trunc (torch.Tensor): The truncated Singular space basis vectors of the predicted trajectory

        Note:
            This function should be called once before training the model.
        """

        ## Training only

        # Trajectory projection
        ## project gt trajectory to Singular space
        C_pred_BK = self.to_Singular_space(pred_traj_norm_BT2, evec_FK=V_pred_trunc_FK).T.detach().cpu().numpy()
        ## KMeans clustering (cluster the trajectories (in the singular space) into N clusters, N: number of samples)
        ## one cluster per sample (20)
        C_anchor_KN = torch.FloatTensor(KMeans(n_clusters=self.s, random_state=0, init='k-means++', n_init=1).fit(C_pred_BK).cluster_centers_.T)

        # Register anchors as model parameters
        self.C_anchor_KN = nn.Parameter(C_anchor_KN.to(self.C_anchor_KN.device))

    def adaptive_anchor_calculation(self, obs_traj_BT2, scene_id_B, vector_field, homography, space):
        r"""Adaptive anchor calculation on Singular space"""

        ## obs_traj_BT2: observed trajectories
        ## scene_id_B: list with the scene names for each pedestrian
        ## homography: dictionary with the homography matrices for each scene
        ## vector_field: dictionary with the vector fields for each scene
        ## space: SingularSpace object

        n_ped = obs_traj_BT2.size(0)
        V_trunc_FK = space.V_trunc_FK

        ## compute the parameters of the normalizer
        space.traj_normalizer.calculate_params(obs_traj_BT2.detach())
        ## expand the initial anchor so that each pedestrian has its own N anchors
        ## anchors are the same for all pedestrians
        init_anchor_BKN = self.C_anchor_KN.unsqueeze(dim=0).repeat_interleave(repeats=n_ped, dim=0).detach()
        init_anchor_NKB = init_anchor_BKN.permute(2, 1, 0)
        ## transform the anchor to Euclidean space
        init_anchor_euclidean_NBT2 = space.batch_to_Euclidean_space(init_anchor_NKB, evec_FK=V_trunc_FK)
        ## denormalize the euclidean space anchor
        ## the denormalization is dependent on the pedestrian,
        ## so essentially it puts the euclidean anchors in front
        ## of the observed trajectories
        init_anchor_euclidean_NBT2 = space.traj_normalizer.denormalize(init_anchor_euclidean_NBT2).cpu().numpy()
        ## work on a copy for creating the adaptive anchor
        adaptive_anchor_euclidean_NBT2 = init_anchor_euclidean_NBT2.copy()
        obs_traj_BT2 = obs_traj_BT2.cpu().numpy()

        ## for each pedestrian
        for ped_id in range(n_ped):
            ## get the scene name (eg. zara1, zara2, ...)
            ## useful for accessing the homography and vector field
            scene_name = scene_id_B[ped_id]

            ## homographies
            hom_image2meters = homography[scene_name]["image2meters"]

            ## transform the initial anchor to image space
            prototype_image_NT2 = world2image(init_anchor_euclidean_NBT2[:, ped_id], hom_image2meters)
            ## transform the last point of the observed trajectory to image space
            startpoint_image_2 = world2image(obs_traj_BT2[ped_id], hom_image2meters)[-1]
            ## get the last point of the prototype (anchor)
            endpoint_image_N2 = prototype_image_NT2[:, -1, :]
            ## round to int (pixel coordinates)
            endpoint_image_N2 = np.round(endpoint_image_N2).astype(int)

            ## original image size (WH)
            size = np.array(vector_field[scene_name].shape[1::-1]) // 2
            ## clip the pixel coordinates of the anchor endpoints to the image size
            endpoint_image_N2 = np.clip(endpoint_image_N2, a_min= -size // 2, a_max=size + size // 2 -1)
            ## for each sample
            for s in range(self.s):
                ## extract from the vector field the coordinates of the nearest valid pixel
                ## to the endpoint of the anchor (in -size//2 to size + size//2 - 1 coordinates)
                vector_2 = np.array(vector_field[scene_name][endpoint_image_N2[s, 1] + size[1] // 2,
                                                           endpoint_image_N2[s, 0] + size[0] // 2])[::-1] - size // 2
                if vector_2[0] == endpoint_image_N2[s, 0] and vector_2[1] == endpoint_image_N2[s, 1]:
                    ## if the nearest pixel is the same as the endpoint, continue
                    ## meaning that the anchor already ends on a valid pixel
                    continue
                else:
                    ## else, the endpoint of the anchor is not a valid pixel (it's over an obstacle)

                    ## store the nearest pixel coordinates of a valid pixel from the endpoint
                    nearest_endpoint_image = vector_2
                    ## compute scale factor for the anchor so that it ends on the nearest valid pixel
                    scale_denom = endpoint_image_N2[s] - startpoint_image_2
                    scale_denom[scale_denom == 0] = 1
                    scale_xy = (nearest_endpoint_image - startpoint_image_2) / scale_denom
                    scale_xy = np.clip(scale_xy, a_min=-1e4, a_max=1e4)

                    ## scale the anchor (prototype) (each point) and store it
                    prototype_image_NT2[s, :, :] = (prototype_image_NT2[s, :, :].copy() - startpoint_image_2) * scale_xy + startpoint_image_2

            ## transform back the prototype to world coordinates
            prototype_world = image2world(prototype_image_NT2, hom_image2meters)
            ## store the adaptive anchor for the current pedestrian
            adaptive_anchor_euclidean_NBT2[:, ped_id] = prototype_world

        ## normalize the adaptive anchor, since the refinement model works in singular space
        ## which requires normalized trajectories (to transform)
        adaptive_anchor_euclidean_NBT2 = space.traj_normalizer.normalize(torch.tensor(adaptive_anchor_euclidean_NBT2, dtype=torch.float32, device=init_anchor_NKB.device))
        ## transform the adaptive anchor to Singular space
        adaptive_anchor_NKB = space.batch_to_Singular_space(adaptive_anchor_euclidean_NBT2, evec_FK=V_trunc_FK)

        adaptive_anchor_BKN = adaptive_anchor_NKB.permute(2, 1, 0).cpu()
        # If you don't want to use an image, return `init_anchor`.
        ## if you don't want to use image information, return the initial anchor
        ## else, return the adaptive anchor
        return adaptive_anchor_BKN

    def forward(self, C_residual, C_anchor):
        r"""Anchor refinement on Singular space

        Args:
            C_residual (torch.Tensor): The predicted Singular space coordinates

        Returns:
            C_pred_refine (torch.Tensor): The refined Singular space coordinates
        """

        C_pred_refine = C_anchor.detach() + C_residual
        return C_pred_refine
