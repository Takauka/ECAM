from collections import defaultdict
import torch


def model_forward_pre_hook(obs_data_KB, obs_ori_2B, addl_info):
    # Pre-process input data for the baseline model
    if obs_ori_2B is not None:
        ## add the (centered) "origins" (last observed point) of the pedestrians
        ## to the embedding of the normalized observed past trajectories
        obs_data_KB = torch.cat([obs_data_KB, obs_ori_2B], dim=0)

    ## get scene mask, number of samples, and anchor from the additional info
    scene_mask_BB = addl_info["scene_mask"]
    num_samples = addl_info["num_samples"]
    anchor_KBN = addl_info["anchor"]
    maps_dict = addl_info["maps"]
    homography_dict = addl_info["homography"]
    scene_ids = addl_info["scene_ids"]
    orig_obs_traj_BT2 = addl_info["original_obs_traj"]

    ## concatenate the observed past trajectories with the anchor,
    ## along the embedding dimension (K)
    obs_data_BK = torch.cat([obs_data_KB.transpose(1, 0), anchor_KBN.permute(1, 2, 0).flatten(start_dim=1)], dim=1)
    obs_data_BK1 = obs_data_BK.unsqueeze(dim=-1)

    ## loc is the anchor (should be a tentative/guess embedding of the future trajectory for each sample of each pedestrian)
    ## in practice the model uses it only to get the output shape
    loc_BNK1 = anchor_KBN.permute(1, 2, 0).unsqueeze(dim=-1)
    input_data = [obs_data_BK1, scene_mask_BB, loc_BNK1, maps_dict, homography_dict, scene_ids, orig_obs_traj_BT2]

    return input_data


def model_forward(input_data, baseline_model):
    # Forward the baseline model with input data
    ## input_data: obs_data_BK1, scene_mask_BB, loc_BNK1
    output_data_BNK1, context_BK, map_patches_B1HW = baseline_model(*input_data)
    ## output_data: BNK1 (K=4 size of the embedding (singular space))
    return output_data_BNK1, context_BK, map_patches_B1HW


def model_forward_post_hook(output_data_BNK1, addl_info=None):
    # Post-process output data of the baseline model
    pred_data_KBN = output_data_BNK1.squeeze(dim=-1).permute(2, 0, 1)

    return pred_data_KBN
