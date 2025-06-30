from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from .anchor import AdaptiveAnchor
from .space import SingularSpace
from . import homography as hm

from baseline.transformerdiffusion.nce.map_nce import MapNceLoss, MapQueryEmbedder, MapKeyEmbedder
import baseline.transformerdiffusion.model_utils as model_utils


class SingularTrajectory(nn.Module):
    r"""The SingularTrajectory model

    Args:
        baseline_model (nn.Module): The baseline model
        hook_func (dict): The bridge functions for the baseline model
        hyper_params (DotDict): The hyper-parameters
    """

    def __init__(self, baseline_model, hook_func, hyper_params, device):
        super().__init__()

        ## diffusion model
        self.baseline_model = baseline_model
        ## bridge functions to connect the SingularTrajectory model with the baseline model
        self.hook_func = hook_func
        self.hyper_params = hyper_params
        ## 8/12 for stochastic, 2/12 for momentary, ...
        self.t_obs, self.t_pred = hyper_params.obs_len, hyper_params.pred_len
        ## whether to use SVD? (always true)
        self.obs_svd, self.pred_svd = hyper_params.obs_svd, hyper_params.pred_svd
        ## embedding space size
        self.k = hyper_params.k
        ## number of samples (maybe s is number of clusters?)
        self.s = hyper_params.num_samples
        ## not sure but it is always 2, so could be the fact that we are working with 2D data
        self.dim = hyper_params.traj_dim
        ## distance threshold to determine whether a trajectory is static
        self.static_dist = hyper_params.static_dist

        self.device = device

        ## Singular space for moving (m) and static (s) pedestrians
        self.Singular_space_m = SingularSpace(hyper_params=hyper_params, norm_sca=True)
        self.Singular_space_s = SingularSpace(hyper_params=hyper_params, norm_sca=False)
        ## Adaptive anchor for moving (m) and static (s) pedestrians
        self.adaptive_anchor_m = AdaptiveAnchor(hyper_params=hyper_params)
        self.adaptive_anchor_s = AdaptiveAnchor(hyper_params=hyper_params)

        #####################
        ## MapNCE
        # Query and key projection heads.
        query_proj = MapQueryEmbedder(256,   ## context size
                                      16)
        event_encoder = MapKeyEmbedder(2, 16)

        self.map_nce = MapNceLoss(obs_len=self.t_obs,
                                  pred_len=self.t_pred,
                                  num_contour_points=10,
                                  query_embedder=query_proj,
                                  key_embedder=event_encoder,
                                  temperature=0.3)


    def calculate_parameters(self, obs_traj_BT2, pred_traj_BT2):
        r"""Generate the Sinuglar space of the SingularTrajectory model

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory

        Note:
            This function should be called once before training the model.
        """

        ## Training only

        # Mask out static trajectory
        mask_B = self.calculate_mask(obs_traj_BT2)
        obs_m_traj_BT2, pred_m_traj_BT2 = obs_traj_BT2[mask_B], pred_traj_BT2[mask_B]
        obs_s_traj_BT2, pred_s_traj_BT2 = obs_traj_BT2[~mask_B], pred_traj_BT2[~mask_B]

        # Descriptor initialization
        data_m = self.Singular_space_m.parameter_initialization(obs_m_traj_BT2, pred_m_traj_BT2)
        data_s = self.Singular_space_s.parameter_initialization(obs_s_traj_BT2, pred_s_traj_BT2)

        ## data_* contains some computed values, so no need to recompute them
        ## data_* = (normalized gt traj, truncated svd matrix)
        ## data_* = (pred_traj_norm_BT2, V_pred_trunc_FK)

        # Anchor initialization
        ## compute the initial anchor by clustering the Singular space coordinates of the GT trajectories
        ## for moving and static pedestrians separately
        self.adaptive_anchor_m.anchor_initialization(*data_m)
        self.adaptive_anchor_s.anchor_initialization(*data_s)

    def calculate_adaptive_anchor(self, dataset):
        obs_traj_BT2, pred_traj_BT2 = dataset.obs_traj_BT2, dataset.pred_traj_BT2
        scene_id_B = dataset.scene_id
        vector_field = dataset.vector_field
        homography = dataset.homography

        obs_traj_BT2 = obs_traj_BT2.to(self.device)
        pred_traj_BT2 = pred_traj_BT2.to(self.device)
        # scene_id_B = scene_id_B.to(self.device)
        # vector_field = vector_field.to(self.device)
        # homography = homography.to(self.device)

        # Mask out static trajectory
        mask_B = self.calculate_mask(obs_traj_BT2)
        mask_cpu_B = mask_B.cpu().numpy()
        obs_m_traj_BT2, scene_id_m_B = obs_traj_BT2[mask_B], scene_id_B[mask_cpu_B]
        obs_s_traj_BT2, scene_id_s_B = obs_traj_BT2[~mask_B], scene_id_B[~mask_cpu_B]

        n_ped = pred_traj_BT2.size(0)
        ## an anchor of size K for each sample of each pedestrian
        anchor_BKN = torch.zeros((n_ped, self.k, self.s), dtype=torch.float)
        ## compute adaptive anchor for moving and static pedestrians
        anchor_BKN[mask_B] = self.adaptive_anchor_m.adaptive_anchor_calculation(obs_m_traj_BT2, scene_id_m_B, vector_field, homography, self.Singular_space_m)
        anchor_BKN[~mask_B] = self.adaptive_anchor_s.adaptive_anchor_calculation(obs_s_traj_BT2, scene_id_s_B, vector_field, homography, self.Singular_space_s)

        return anchor_BKN

    def calculate_mask(self, obs_traj_BT2):
        if obs_traj_BT2.size(1) <= 2:
            mask_B = (obs_traj_BT2[:, -1] - obs_traj_BT2[:, -2]).div(1).norm(p=2, dim=-1) > self.static_dist
        else:
            ## divide by 2 since considering point -1 and -3
            mask_B = (obs_traj_BT2[:, -1] - obs_traj_BT2[:, -3]).div(2).norm(p=2, dim=-1) > self.static_dist
        return mask_B

    def forward(self, obs_traj_BT2, adaptive_anchor_BKN, pred_traj_BT2=None, addl_info=None):
        r"""The forward function of the SingularTrajectory model

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory (GT) (optional, for training only)
            addl_info (dict): The additional information (optional, if baseline model requires)

        Returns:
            output (dict): The output of the model (recon_traj, loss, etc.)
        """

        n_ped = obs_traj_BT2.size(0)

        # Filter out static trajectory
        mask_B = self.calculate_mask(obs_traj_BT2)
        obs_m_traj_BT2 = obs_traj_BT2[mask_B]       ## moving pedestrians
        obs_s_traj_BT2 = obs_traj_BT2[~mask_B]      ## static pedestrians
        pred_m_traj_gt_BT2 = pred_traj_BT2[mask_B] if pred_traj_BT2 is not None else None
        pred_s_traj_gt_BT2 = pred_traj_BT2[~mask_B] if pred_traj_BT2 is not None else None

        # Projection
        ## Project the observed past trajs, and the future gt trajs to the Singular space
        ## project separately the moving and static pedestrians
        ## projection also performs the normalization of the trajectories before projecting them
        C_m_obs_KB, C_m_pred_gt_KB = self.Singular_space_m.projection(obs_m_traj_BT2, pred_m_traj_gt_BT2)
        C_s_obs_KB, C_s_pred_gt_KB = self.Singular_space_s.projection(obs_s_traj_BT2, pred_s_traj_gt_BT2)
        ## merge again the moving and static pedestrians in a single tensor
        C_obs_KB = torch.zeros((self.k, n_ped), dtype=torch.float, device=obs_traj_BT2.device)
        C_obs_KB[:, mask_B], C_obs_KB[:, ~mask_B] = C_m_obs_KB, C_s_obs_KB

        # Absolute coordinate
        ## get "origin" of the observed past trajs (origin = last observed point)
        obs_m_ori_2B = self.Singular_space_m.traj_normalizer.traj_ori_B12.squeeze(dim=1).T
        obs_s_ori_2B = self.Singular_space_s.traj_normalizer.traj_ori_B12.squeeze(dim=1).T
        ## merge the origins of moving and static pedestrians in a single tensor
        obs_ori_2B = torch.zeros((2, n_ped), dtype=torch.float, device=obs_traj_BT2.device)
        obs_ori_2B[:, mask_B], obs_ori_2B[:, ~mask_B] = obs_m_ori_2B, obs_s_ori_2B
        ## center the observed past trajs origins
        ## substract the mean origin of the pedestrians in the scene from
        ## the origin of each pedestrian, so that the mean origin of the scene is 0
        obs_ori_2B -= obs_ori_2B.mean(dim=1, keepdim=True)

        # Adaptive anchor per agent
        C_anchor_KBN = adaptive_anchor_BKN.permute(1, 0, 2)
        ## save the anchor in the additional info
        addl_info["anchor"] = C_anchor_KBN.clone()

        ## save the observed past trajs and the origins in the additional info
        addl_info["original_obs_traj"] = obs_traj_BT2


        # Trajectory prediction
        ## forward pass of the baseline model
        ## the hook functions are used to connect the SingularTrajectory model with the baseline model
        ## the hook functions are used to preprocess the input data, and postprocess the output data
        input_data = self.hook_func.model_forward_pre_hook(C_obs_KB, obs_ori_2B, addl_info)
        output_data_BNK1, context_BK, map_patches_B1HW = self.hook_func.model_forward(input_data, self.baseline_model)
        ## The output of the model are the residuals of the Singular space coordinates
        ## that are used to refine the adaptive anchor (still on the Singular space)
        C_pred_refine_KBN = self.hook_func.model_forward_post_hook(output_data_BNK1, addl_info) * 0.1

        ## Refine the anchors with the predicted residuals (from the diffusion model)
        ## still in the Singular space (K=4).
        ## Moving/static pedestrians are refined separately.
        ## At the end of the day is just a summation.
        C_m_pred_KBN = self.adaptive_anchor_m(C_pred_refine_KBN[:, mask_B], C_anchor_KBN[:, mask_B])
        C_s_pred_KBN = self.adaptive_anchor_s(C_pred_refine_KBN[:, ~mask_B], C_anchor_KBN[:, ~mask_B])

        # Reconstruction
        ## reconstruct the predicted Singular space coordinates (vector of size K = 4)
        ## to the Euclidean space (vector of size T_pred * 2 -> Tx2)
        pred_m_traj_recon_NBT2 = self.Singular_space_m.reconstruction(C_m_pred_KBN)
        pred_s_traj_recon_NBT2 = self.Singular_space_s.reconstruction(C_s_pred_KBN)
        ## merge back the moving and static pedestrians in a single tensor
        pred_traj_recon_NBT2 = torch.zeros((self.s, n_ped, self.t_pred, self.dim), dtype=torch.float, device=obs_traj_BT2.device)
        pred_traj_recon_NBT2[:, mask_B], pred_traj_recon_NBT2[:, ~mask_B] = pred_m_traj_recon_NBT2, pred_s_traj_recon_NBT2

        ## build the output dictionary
        output = {"recon_traj": pred_traj_recon_NBT2}

        ## compute the loss if the GT trajectories are provided (training)
        if pred_traj_BT2 is not None:
            ## build the merged tensor of the singular space
            C_pred_KBN = torch.zeros((self.k, n_ped, self.s), dtype=torch.float, device=obs_traj_BT2.device)
            C_pred_KBN[:, mask_B], C_pred_KBN[:, ~mask_B] = C_m_pred_KBN, C_s_pred_KBN

            # Low-rank approximation for gt trajectory
            ## merge back the moving and static ground truth low rank (Singular space) trajectories
            C_pred_gt_KB = torch.zeros((self.k, n_ped), dtype=torch.float, device=obs_traj_BT2.device)
            C_pred_gt_KB[:, mask_B], C_pred_gt_KB[:, ~mask_B] = C_m_pred_gt_KB, C_s_pred_gt_KB
            ## detach the gt low rank trajectories, since it is sufficient to
            ## pull the predicted euclidean trajectories close to the gt euclidean trajectories.
            ## the low rank approximation will become close by itself.
            C_pred_gt_KB = C_pred_gt_KB.detach()

            # Loss calculation
            ## error in the Singular space (low rank coefficients)
            error_coefficient = (C_pred_KBN - C_pred_gt_KB.unsqueeze(dim=-1)).norm(p=2, dim=0)
            ## error in the Euclidean space (trajectory displacement)
            error_displacement_NBT = (pred_traj_recon_NBT2 - pred_traj_BT2.unsqueeze(dim=0)).norm(p=2, dim=-1)
            ## consider just the min error for each pedestrian
            output["loss_eigentraj"] = error_coefficient.min(dim=-1)[0].mean()
            output["loss_euclidean_ade"] = error_displacement_NBT.mean(dim=-1).min(dim=0)[0].mean()
            output["loss_euclidean_fde"] = error_displacement_NBT[:, :, -1].min(dim=0)[0].mean()

            ## compute the diversity loss
            # diversity_loss = self.compute_angle_diversity_loss(pred_traj_recon_NBT2, pred_traj_BT2, obs_traj_BT2[:, -1])
            # output["loss_diversity"] = diversity_loss
            output["loss_diversity"] = torch.tensor(0.0, device=obs_traj_BT2.device)

            ## MapNCE loss

            ## full trajectory (observed + predicted)
            traj_BT2 = torch.cat([obs_traj_BT2, pred_traj_BT2], dim=1)

            if self.hyper_params.baseline_use_map:
                map_nce_loss = self.map_nce(traj_BT2,
                                            context_BK,         # map dependent context
                                            map_patches_B1HW)

                output["loss_map_nce"] = map_nce_loss

                ## Environment collision loss

                env_collision_loss_total = 0
                ## for each different scene
                scene_ids_B = addl_info["scene_ids"]
                maps_dict = addl_info["maps"]
                homography_dict = addl_info["homography"]
                vector_field_dict = addl_info["vector_field"]
                for dataset_name in maps_dict.keys():
                    map_mask_1HW = (maps_dict[dataset_name]).to(device=obs_traj_BT2.device) * 255
                    hom_meters2mask = torch.from_numpy(homography_dict[dataset_name]["meters2mask"]).to(obs_traj_BT2.device)
                    hom_meters2image = torch.from_numpy(homography_dict[dataset_name]["meters2image"]).to(obs_traj_BT2.device)
                    hom_image2meters = torch.from_numpy(homography_dict[dataset_name]["image2meters"]).to(obs_traj_BT2.device)
                    vector_field = torch.from_numpy(vector_field_dict[dataset_name]).to(obs_traj_BT2.device)
                    img_size = torch.tensor(vector_field.shape[1::-1], device=obs_traj_BT2.device) // 2

                    ## get prediction for the scene
                    scene_pred_BT2 = pred_traj_BT2[scene_ids_B == dataset_name]
                    scene_pred_hat_NBT2 = pred_traj_recon_NBT2[:, scene_ids_B == dataset_name]
                    scene_pred_hat_AT2 = scene_pred_hat_NBT2.view(-1, self.t_pred, 2)

                    # If there are no predictions for the scene, skip it.
                    if scene_pred_BT2.size(0) == 0:
                        continue

                    mode = self.hyper_params.env_col_loss_mode

                    env_collision_loss = self.compute_env_col_loss(scene_pred_BT2, scene_pred_hat_AT2, map_mask_1HW, hom_meters2mask, hom_meters2image, hom_image2meters, img_size, vector_field, mode=mode)
                    env_collision_loss_total += env_collision_loss

                output["loss_env_collision"] = env_collision_loss_total

            else:
                output["loss_map_nce"] = torch.tensor(0.0, device=obs_traj_BT2.device)
                output["loss_env_collision"] = torch.tensor(0.0, device=obs_traj_BT2.device)

        return output


    @torch.no_grad()
    def generate_artificial_gt(self, scene_pred_hat_AT2, vector_field, map_mask_1HW, hom_meters2mask, hom_meters2image, hom_image2meters, img_size, min_margin, max_margin):
        """Generate artificial ground truth for the colliding trajectories.

        Returns:
            artificial_gt_AT2: Artificial ground truth for the colliding trajectories.
            valid_env_gt_collisions_A: Boolean mask indicating which of the generated
                artificial ground truth trajectories are valid (not over an obstacle).
        """

        # Compute artificial GTs for the colliding trajectories,
        # using the vector field.

        # Project all trajectories to image space.
        traj_image_AT2 = hm.project(scene_pred_hat_AT2, hom_meters2image).int()

        # Clamp the trajectory coordinates.
        traj_image_AT2 = torch.clamp(traj_image_AT2,
                                     min=-img_size//2,
                                     max=img_size + img_size//2 - 1)

        # Calculate indices for vector field sampling.
        idx_h = traj_image_AT2[:, :, 1] + img_size[1] // 2
        idx_w = traj_image_AT2[:, :, 0] + img_size[0] // 2

        # Sample the vector field for all trajectories.
        closest_valid_pos_img_AT2 = vector_field[idx_h, idx_w]

        # Flip the second dimension and adjust coordinates.
        closest_valid_pos_img_AT2 = closest_valid_pos_img_AT2.flip(2) - img_size // 2

        # Project back to meters.
        closest_valid_pos_AT2 = hm.project(closest_valid_pos_img_AT2, hom_image2meters)

        # Compute the displacement vectors.
        displ_AT2 = (closest_valid_pos_AT2 - scene_pred_hat_AT2).float()
        # Compute the normalized direction vectors.
        norm_AT1 = displ_AT2.norm(p=2, dim=-1, keepdim=True)
        almost_zero = torch.isclose(norm_AT1, torch.zeros_like(norm_AT1), atol=1e-1)
        norm_AT1[almost_zero] = 1
        dir_AT2 = displ_AT2 / norm_AT1
        dir_AT2[almost_zero.expand_as(dir_AT2)] = 0

        # Compute the artificial ground truth.
        artificial_gt_AT2 = (scene_pred_hat_AT2 + displ_AT2).float()

        # Margin from the obstacle (in meters).
        margin_range = max_margin - min_margin
        margin_A = torch.rand(scene_pred_hat_AT2.size(0), device=scene_pred_hat_AT2.device) * margin_range + min_margin
        margin_AT2 = dir_AT2 * margin_A[:, None, None]
        artificial_gt_AT2 = artificial_gt_AT2 + margin_AT2

        # Augment the trajectory resolution and verify that
        # no point is over an obstacle.
        artificial_gt_aug_AT2 = model_utils.augment_traj_resolution(
            artificial_gt_AT2, parts=1
        )

        # Check for collisions.
        env_gt_collisions_A = model_utils.check_env_collisions(
            artificial_gt_aug_AT2,
            map_mask_1HW,
            torch.eye(3).to(artificial_gt_AT2.device),
            hom_meters2mask
        )

        valid_env_gt_collisions_A = ~env_gt_collisions_A

        return artificial_gt_AT2, valid_env_gt_collisions_A



    def compute_env_col_loss(self, scene_pred_BT2, scene_pred_hat_AT2, map_mask_1HW, hom_meters2mask, hom_meters2image, hom_image2meters, img_size, vector_field, mode: Literal["true-gt", "synth-gt"]):
        env_collisions_AP = model_utils.check_env_collisions_precise(
            scene_pred_hat_AT2,
            map_mask_1HW,
            torch.eye(3).to(scene_pred_hat_AT2.device),
            hom_meters2mask
        )

        if mode == "synth-gt":
            env_collisions_A = env_collisions_AP.any(dim=-1)
            up_to_first_col_included_AP = env_collisions_AP.cumsum(dim=-1) <= 1

            # Artificial GTs generation.
            min_margin = self.hyper_params.env_col_loss_synth_gt_min_margin
            max_margin = self.hyper_params.env_col_loss_synth_gt_max_margin
            artificial_gt_AT2, valid_env_gt_collisions_A = self.generate_artificial_gt(
                scene_pred_hat_AT2, vector_field, map_mask_1HW, hom_meters2mask, hom_meters2image, hom_image2meters, img_size, min_margin=min_margin, max_margin=max_margin
            )

            # Want to apply loss to all the trajectories that collide with the environment,
            # but only if the artificial GTs are valid (not over an obstacle).
            fake_env_collisions_AP = env_collisions_A.unsqueeze(dim=-1).expand_as(env_collisions_AP)

            loss_mask_AT = fake_env_collisions_AP & up_to_first_col_included_AP

            gt_C2 = artificial_gt_AT2[loss_mask_AT]

            # Compute the loss for the colliding trajectories (if any).
            env_collision_loss = torch.tensor(0.0, device=scene_pred_hat_AT2.device)
            if scene_pred_hat_AT2[loss_mask_AT].size(0) > 0:
                env_collision_loss = \
                F.mse_loss(scene_pred_hat_AT2[loss_mask_AT],
                           gt_C2,
                           reduction='mean')

                if torch.isnan(env_collision_loss):
                    env_collision_loss = torch.tensor(0.0, device=env_collision_loss.device)


        else:       # mode == "true-gt"
            env_collisions_A = env_collisions_AP.any(dim=-1)

            env_collisions_NB = env_collisions_A.view(self.s, -1)
            loss_mask_A = env_collisions_A

            # Make the colliding trajectories be part of the loss.
            # Get the index of the ground truth trajectory for each
            # trajectory that collides with the environment.
            # Shape: (num_collisions,) where num_collisions <= A
            _, gt_index_A = torch.where(env_collisions_NB)
            # Get the ground truth trajectory for each colliding trajectory.
            # Shape: (num_collisions, pred_len, 2)
            gt_AT2 = scene_pred_BT2[gt_index_A]

            # Compute the loss for the colliding trajectories (if any).
            env_collision_loss = torch.tensor(0.0, device=scene_pred_hat_AT2.device)
            if scene_pred_hat_AT2[loss_mask_A].size(0) > 0:
                env_collision_loss = \
                F.mse_loss(scene_pred_hat_AT2[loss_mask_A],
                           gt_AT2,
                           reduction='mean')

                if torch.isnan(env_collision_loss):
                    env_collision_loss = torch.tensor(0.0, device=env_collision_loss.device)

        return env_collision_loss

    def compute_angle_diversity_loss(self, scene_pred_hat_NBT2, pred_traj_BT2, last_obs_B2):
        # Compute the diversity loss.

        # Make the model predict trajectories in all possible directions.

        # Permute the dimensions to have the batch dimension first.
        scene_pred_hat_BNT2 = scene_pred_hat_NBT2.permute(1, 0, 2, 3)

        # Compute direction vectors for all the predicted trajectories.
        # Shape: (num_pedestrians, num_samples, 2)
        dir_BN2 = scene_pred_hat_BNT2[:, :, -1] - last_obs_B2[:, None]
        dir_norm_BN1 = dir_BN2.norm(p=2, dim=-1, keepdim=True)

        # Drop the trajectory with the closest final point to the last observed point.
        # Shape: (num_pedestrians, num_samples - 1, 2)
        batch_size = dir_BN2.size(0)
        closest_index_B = dir_norm_BN1.argmin(dim=1).squeeze(dim=-1)
        closest_index_B12 = closest_index_B[:, None, None].expand(-1, -1, 2)
        keep_mask_BN2 = torch.ones_like(dir_BN2).scatter_(1, closest_index_B12, 0).bool()
        dir_BN2 = dir_BN2[keep_mask_BN2].view(batch_size, -1, dir_BN2.size(-1))
        dir_norm_BN1 = dir_norm_BN1[keep_mask_BN2[:, :, 1:]].view(batch_size, -1, dir_norm_BN1.size(-1))

        # Normalize the direction vectors.
        dir_BN2 = dir_BN2 / torch.clamp(dir_norm_BN1, min=1e-6)

        # Compute the pairwise dot products between all the direction vectors.
        # Shape: (num_pedestrians, num_samples - 1, num_samples - 1)
        dot_BNN = torch.bmm(dir_BN2, dir_BN2.transpose(1, 2))

        EPS = 1e-6
        angles_BNN = torch.acos(dot_BNN.clamp(min=-1 + EPS, max=1 - EPS))

        # Compute the diversity loss.
        # The diversity loss is the sum of the angles between all the direction vectors.
        diversity_loss = angles_BNN.sum(dim=(1, 2)).mean()

        return -diversity_loss

    def compute_end_diversity_loss(self, scene_pred_hat_NBT2):
        # Compute the diversity loss.
        # The diversity loss is the sum of the pairwise distances between
        # the final points of the trajectories sampled for a pedestrian.

        # Permute the dimensions to have the batch dimension first.
        scene_pred_hat_BNT2 = scene_pred_hat_NBT2.permute(1, 0, 2, 3)

        # Compute the pairwise distances between all the sampled trajectories.
        # Shape: (num_pedestrians, num_samples, num_samples)
        pairwise_dist_BNN = torch.cdist(scene_pred_hat_BNT2[:, :, -1], scene_pred_hat_BNT2[:, :, -1], p=2)

        # Compute the diversity loss.
        # The diversity loss is the sum of the pairwise distances between
        # the final points of the trajectories sampled for a pedestrian.
        diversity_loss = pairwise_dist_BNN.sum(dim=(1, 2)).mean()

        return -diversity_loss

    @staticmethod
    def __debug_artificial_gt(scene_pred_hat_NBT2, map_mask_1HW, hom_meters2mask, env_collisions_NB, artificial_gt_AT2, valid_env_gt_collisions_A):
        # Visualize the artificial GTs.
        import matplotlib.pyplot as plt

        shape = scene_pred_hat_NBT2.shape
        artificial_gt_NBT2 = artificial_gt_AT2.view(shape)
        artificial_gt_BNT2 = artificial_gt_NBT2.permute(1, 0, 2, 3)
        scene_pred_hat_BNT2 = scene_pred_hat_NBT2.permute(1, 0, 2, 3)
        valid_env_gt_collisions_NB = valid_env_gt_collisions_A.view(self.s, -1)
        valid_env_gt_collisions_BN = valid_env_gt_collisions_NB.permute(1, 0)
        env_collisions_BN = env_collisions_NB.permute(1, 0)
        for gt_NT2, pred_NT2, valid_gt_N, env_col_N  in zip(artificial_gt_BNT2, scene_pred_hat_BNT2, valid_env_gt_collisions_BN, env_collisions_BN):
            # plot
            fig, axs = plt.subplots(2, 1, figsize=(10, 10))
            # artificial GTs
            gt_mask_NT2 = hm.project(gt_NT2, hom_meters2mask).int()
            axs[0].imshow(map_mask_1HW.squeeze().cpu().numpy(), cmap='gray')
            for gt_T2, valid in zip(gt_mask_NT2, valid_gt_N):
                color = 'g' if valid else 'r'
                axs[0].plot(gt_T2[:, 0].cpu().numpy(),
                            gt_T2[:, 1].cpu().numpy(), color)
            # original predictions
            pred_mask_NT2 = hm.project(pred_NT2, hom_meters2mask).int()
            axs[1].imshow(map_mask_1HW.squeeze().cpu().numpy(), cmap='gray')
            for pred_T2, env_col in zip(pred_mask_NT2, env_col_N):
                color = 'r' if env_col else 'g'
                axs[1].plot(pred_T2[:, 0].cpu().numpy(),
                            pred_T2[:, 1].cpu().numpy(), color)

            plt.show()

