import math
import torch
import torch.nn as nn
from torch.nn import Module, Linear
import numpy as np
from .layers import PositionalEncoding, ConcatSquashLinear

from baseline.transformerdiffusion.mask_autoenc.mask_autoencoder import PatchEncoder
import baseline.transformerdiffusion.model_utils as model_utils


class st_encoder(nn.Module):
    """Transformer Denoising Model
    codebase borrowed from https://github.com/MediaBrain-SJTU/LED"""
    def __init__(self):
        super().__init__()
        channel_in = 2
        channel_out = 32
        dim_kernel = 3
        self.dim_embedding_key = 256
        self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.temporal_encoder = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.spatial_conv.weight)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
        nn.init.zeros_(self.spatial_conv.bias)
        nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
        nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

    def forward(self, X):
        X_t = torch.transpose(X, 1, 2)
        X_after_spatial = self.relu(self.spatial_conv(X_t))
        X_embed = torch.transpose(X_after_spatial, 1, 2)
        output_x, state_x = self.temporal_encoder(X_embed)
        state_x = state_x.squeeze(0)
        return state_x


class social_transformer(nn.Module):
    """Transformer Denoising Model
    codebase borrowed from https://github.com/MediaBrain-SJTU/LED"""
    def __init__(self, cfg, additional_dim):
        super(social_transformer, self).__init__()
        ## cfg.k: embedding size
        ## cfg.s: number of samples to generate

        ## Linear transformation of the embedding of a pedestrian.
        ## The embedding is the concatenation of the past trajectory embedding,
        ## the origin, and the adaptive anchors.

        ## Note that the s samples are mixed together in the same embedding space,
        ## so the each sample influences the generation of each other sample.

        ## cfg.k * cfg.s: due to the s (=N) anchors of embedding size k
        ## + cfg.k: due to the embedding size of the past trajectory (k)
        ## + 2: due to the origin (2d coord)
        self.encode_past = nn.Linear(cfg.k*cfg.s+cfg.k+2 + additional_dim, 256, bias=False)

        ## define TransformerEncoder
        self.layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)

    def forward(self, h_BK1, mask_BB):
        ## apply linear layer
        h_feat_B1K = self.encode_past(h_BK1.reshape(h_BK1.size(0), -1)).unsqueeze(1)
        ## apply transformer encoder
        ## a token is a person's embedding (past_traj_emb + origin + adaptive anchors)
        ## 1 is the batch dimension for the transformer encoder
        ## B is the temporal dimension for the transformer encoder
        ## K is the embedding dimension for the transformer encoder
        ## pass directly the mask as attention mask
        h_feat__B1K = self.transformer_encoder(h_feat_B1K, mask_BB)
        ## residual connection
        h_feat_B1K = h_feat_B1K + h_feat__B1K

        return h_feat_B1K


class TransformerDenoisingModel(Module):
    """Transformer Denoising Model
    codebase borrowed from https://github.com/MediaBrain-SJTU/LED"""
    def __init__(self, context_dim=256, cfg=None):
        super().__init__()

        assert cfg is not None, "cfg must be provided"

        if cfg.baseline_use_map:
            MAP_EMB_DIM = 32
        else:
            MAP_EMB_DIM = 0

        self.context_dim = context_dim
        ## spatial dimension is set to 1, likely because we work on a 1d embedding vector
        self.spatial_dim = 1
        ## use the embedding dimension K as the size of the temporal dimension.
        ## likely a sort of artifact of how the model was implemented by the original authors
        ## for the original tasks (also valid for spatial_dim)
        ## At least k should be the size of the final embedding vector
        ## that will be later remapped to the original time dimension.
        self.temporal_dim = cfg.k
        ## number of samples to generate
        self.n_samples = cfg.s
        ## context encoder model (likely encodes the social context)

        self.encoder_context = social_transformer(cfg, MAP_EMB_DIM)

        if cfg.baseline_use_map:
            ## map patch encoder
            self.map_encoder = PatchEncoder(64)
            # self.map_encoder.requires_grad_(False)
            checkpoint = torch.load("checkpoints/patch_enc_ps.ckpt", map_location="cpu")
            encoder_weights = {k: v for k, v in checkpoint["state_dict"].items()
                               if k.startswith("autoencoder.encoder.")}
            renamed_weights = {
                k.replace("autoencoder.encoder.", ""): v
                for k, v in encoder_weights.items()
            }
            self.map_encoder.load_state_dict(renamed_weights)
            self.map_encoder.requires_grad_(False)
            self.map_mlp = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, MAP_EMB_DIM)
            )
        ##

        ## positional encoding (sine and cosine)
        ## 24 = pred_len=12 x coord_size=2
        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
        ## linear layers conditioned on context (additional gating and bias)
        ## this sequence of layers starts from (n_samples * spatial_dim * temporal_dim) and
        ## ends with the same (n_samples * spatial_dim * temporal_dim)
        self.concat1 = ConcatSquashLinear(self.n_samples*self.spatial_dim*self.temporal_dim, 2*context_dim, context_dim+3)
        self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
        self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
        self.linear = ConcatSquashLinear(context_dim//2, self.n_samples*self.spatial_dim*self.temporal_dim, context_dim+3)
        # self.concat1 = ConcatSquashLinear(self.n_samples*self.spatial_dim*self.temporal_dim, 2*context_dim, context_dim+3 + MAP_EMB_DIM)
        # self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3 + MAP_EMB_DIM)
        # self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3 + MAP_EMB_DIM)
        # self.linear = ConcatSquashLinear(context_dim//2, self.n_samples*self.spatial_dim*self.temporal_dim, context_dim+3 + MAP_EMB_DIM)

    def forward(self, x, beta, context, mask):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        ctx_emb = torch.cat([time_emb, context], dim=-1)
        x = self.concat1(ctx_emb, x)
        final_emb_XBX = x.permute(1,0,2)
        final_emb_XBX = self.pos_emb(final_emb_XBX)
        trans_BXX = self.transformer_encoder(final_emb_XBX).permute(1,0,2)
        trans_BXX = self.concat3(ctx_emb, trans_BXX)
        trans_BXX = self.concat4(ctx_emb, trans_BXX)
        return self.linear(ctx_emb, trans_BXX)

    def encode_context(self, context_B1K, mask_BB):
        ## fill the mask with -inf and 0
        mask_BB = mask_BB.float().masked_fill(mask_BB == 0, float('-inf')).masked_fill(mask_BB == 1, float(0.0))
        ## forward through "social_transformer"
        context_B1K = self.encoder_context(context_B1K, mask_BB)
        return context_B1K

    def generate_accelerate(self, x_BNK1, beta_B111, context_B1K, map_emb_BK, mask_BB):
        beta_B1 = beta_B111.view(beta_B111.size(0), 1)
        time_emb_B3 = torch.cat([beta_B1, torch.sin(beta_B1), torch.cos(beta_B1)], dim=-1)

        ## append time embedding to context embedding
        ## append map embedding to context embedding
        ## K = 256*1 + 3 + 32
        ctx_emb_BK = torch.cat([time_emb_B3,
                                context_B1K.view(-1, self.context_dim*self.spatial_dim),
                                # map_emb_BK
                               ], dim=-1)

        ## forward through the model
        ## batch_generate is equal to forward
        ## the following layers are a sort of gated linear layers conditioned on the context
        ## that are applied to the input x_BNK1
        ## input x_BNK1 is the vector the model is trying to denoise
        ## context_BK is the social context, time embedding, ...
        ## Note that the samples are mixed together in the same embedding space,
        ## so the each sample influences the generation of each other sample.
        ## DEPENDENT samples instead of independent samples.
        ## C: channel dimension
        trans_BC = self.concat1.batch_generate(ctx_emb_BK, x_BNK1.view(-1, self.n_samples*self.temporal_dim*self.spatial_dim))
        trans_BC = self.concat3.batch_generate(ctx_emb_BK, trans_BC)
        trans_BC = self.concat4.batch_generate(ctx_emb_BK, trans_BC)
        ## return the denoised vector
        ## BNK1
        return self.linear.batch_generate(ctx_emb_BK, trans_BC).view(-1, self.n_samples, self.temporal_dim, self.spatial_dim)

    def encode_maps(self, map_B1HW):
        ## forward through the map encoder
        map_emb_BK = self.map_encoder(map_B1HW)
        map_emb_BK = self.map_mlp(map_emb_BK)
        return map_emb_BK


class DiffusionModel(Module):
    """Transformer Denoising Model
    codebase borrowed from https://github.com/MediaBrain-SJTU/LED"""
    def __init__(self, cfg):
        super().__init__()
        ## cfg is a dictionary containing mostly hyperparameters
        ## related to diffusion process, and also traj embedding dim (K)
        ## and number of samples to generate
        self.cfg = cfg
        ## actual denoising model
        self.model = TransformerDenoisingModel(context_dim=256, cfg=cfg)

        ## used params
        # 'scheduler': 'ddim'
        # 'steps': 10
        # 'beta_start': 1.e-4
        # 'beta_end': 5.e-2
        # 'beta_schedule': 'linear'
        # 'k': hyper_params.k
        # 's': hyper_params.num_samples


        ## D: number of denoising steps

        # self.betas_D = self.make_beta_schedule(
        #     schedule=self.cfg.beta_schedule, n_timesteps=self.cfg.steps,
        #     start=self.cfg.beta_start, end=self.cfg.beta_end).to(device)

        # self.alphas_D = 1 - self.betas_D
        # self.alphas_prod = torch.cumprod(self.alphas_D, 0)
        # self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        # self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)

        self.register_buffer(
            "betas_D",
            self.make_beta_schedule(
                schedule=self.cfg.beta_schedule,
                n_timesteps=self.cfg.steps,
                start=self.cfg.beta_start,
                end=self.cfg.beta_end
            )
        )

        self.register_buffer("alphas_D", 1 - self.betas_D)
        self.register_buffer("alphas_prod", torch.cumprod(self.alphas_D, dim=0))
        self.register_buffer("alphas_bar_sqrt", torch.sqrt(self.alphas_prod))
        self.register_buffer("one_minus_alphas_bar_sqrt", torch.sqrt(1 - self.alphas_prod))


    def make_beta_schedule(self, schedule: str = 'linear',
            n_timesteps: int = 1000,
            start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        return betas

    def extract(self, input_D, t_B, x_BNK1):
        ## this function is more general than the reported shapes

        shape = x_BNK1.shape
        out_D = torch.gather(input_D, 0, t_B.to(input_D.device))
        reshape = [t_B.shape[0]] + [1] * (len(shape) - 1)

        ## output: B x 1 x 1 x 1
        return out_D.reshape(*reshape)

    def forward(self, past_traj_BK1, traj_mask_BB, loc_BNK1, maps_dict, homography_dict,
                scene_ids, orig_obs_traj_BT2):
        ## past_traj_BK1: obs_traj_sing, obs_traj_origin, anchor_sing_1, ..., anchor_sing_20
        ## traj_mask_BB: scene mask that indicates which pedestrians are present in the scene
        ## loc_BNK1: anchor_sing_1, ..., anchor_sing_20

        ## maps_dict: dictionary with the map for each scene
        ## scene_ids: list with the scene names for each pedestrian
        ## orig_obs_traj_BT2: observed trajectories (non-normalized)

        ## extract patches from the maps.

        map_masks = [ maps_dict[scene_ids[i]] * 255 for i in range(len(scene_ids)) ]

        # Find maximum dimensions across all map masks
        max_h = max(mask.shape[1] for mask in map_masks)
        max_w = max(mask.shape[2] for mask in map_masks)

        # Pad and stack map masks
        padded_masks = []
        for mask in map_masks:
            h_pad = max_h - mask.shape[1]
            w_pad = max_w - mask.shape[2]
            padded_mask = torch.nn.functional.pad(mask, (0, w_pad, 0, h_pad), value=0)
            padded_masks.append(padded_mask)

        map_masks_B1HW = torch.stack(padded_masks, dim=0)

        ped_traj_BT2 = orig_obs_traj_BT2

        scene_transform_matrix_B33 = torch.eye(3, device=orig_obs_traj_BT2.device).\
            unsqueeze(0).\
            expand(ped_traj_BT2.shape[0], -1, -1)

        hom_meters2mask_B33 = torch.stack([torch.from_numpy(homography_dict[scene_id]["meters2mask"]).to(orig_obs_traj_BT2.device)
                                           for scene_id in scene_ids], dim=0)


        patch_B1HW, img_pos_B2, img_prev_pos_B2 = \
            model_utils.extract_patches_batched(ped_traj_BT2,
                                                map_masks_B1HW,
                                                scene_transform_matrix_B33,
                                                hom_meters2mask_B33,
                                                patch_size_px=100,
                                                back_dist_px=10)

        if self.cfg.baseline_use_map:
            ## list to Tensor
            # map_B1HW = torch.cat(patch_list, dim=0)
            map_B1HW = patch_B1HW

            ## encode the patches
            map_emb_BK = self.model.encode_maps(map_B1HW)
        else:
            ## Empty tensor
            map_B1HW = None
            map_emb_BK = torch.zeros(loc_BNK1.shape[0], 0, device=loc_BNK1.device)

        pred_traj_BNK1, context_BK = self.p_sample_forward(past_traj_BK1, traj_mask_BB, loc_BNK1, map_emb_BK)
        return pred_traj_BNK1, context_BK, map_B1HW

    def p_sample(self, x_BK1, mask_BB, cur_y_BNK1, t, context_B1K, map_emb_BK):
        t = torch.tensor([t], device=x_BK1.device)
        ## essentially get the beta for this time step and
        ## reshape it so it matches the shape of cur_y_BNK1 (broadcasting)
        beta_B111 = self.extract(self.betas_D, t.repeat(x_BK1.shape[0]), cur_y_BNK1)
        ## actual denoising step (compute noise using model)
        eps_theta = self.model.generate_accelerate(cur_y_BNK1, beta_B111, context_B1K, map_emb_BK, mask_BB)
        ## diffusion formula
        eps_factor = ((1 - self.extract(self.alphas_D, t, cur_y_BNK1)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y_BNK1))
        mean = (1 / self.extract(self.alphas_D, t, cur_y_BNK1).sqrt()) * (cur_y_BNK1 - (eps_factor * eps_theta))

        ## sample z from normal distribution (sampling trick)
        # Fix the random seed for reproducibility
        if False:
            z_BNK1 = torch.randn_like(cur_y_BNK1).to(x_BK1.device)
        else:
            rng = torch.Generator(device=x_BK1.device)
            rng.manual_seed(0)
            z_BNK1 = torch.normal(mean=0.0, std=1.0, size=cur_y_BNK1.shape, generator=rng, device=x_BK1.device)

        sigma_t = self.extract(self.betas_D, t, cur_y_BNK1).sqrt()
        ## rescale
        sample_BNK1 = mean + sigma_t * z_BNK1 * 0.00001
        return (sample_BNK1)

    def p_sample_forward(self, x_BK1, mask_BB, loc_BNK1, map_emb_BK):
        ## loc_BNK1: (anchor_sing_1, ..., anchor_sing_20) for each pedestrian
        ## loc_BNK1 used only to get the output shape

        ## sample initial noise (normal distribution with mean 0 and std 1)
        ## that will be iteratively denoised.
        # Fix the random seed for reproducibility
        if False:
            cur_y_BNK1 = torch.randn_like(loc_BNK1)
        else:
            rng = torch.Generator(device=x_BK1.device)
            rng.manual_seed(0)
            cur_y_BNK1 = torch.normal(mean=0.0, std=1.0, size=loc_BNK1.shape, generator=rng, device=x_BK1.device)

        ## context is composed by
        ## past_traj_BK1: obs_traj_sing, obs_traj_origin, anchor_sing_1, ..., anchor_sing_20
        ## (and depends on the scene mask)
        ## It captures (among other things?) the social context
        # context_BK1 = self.model.encode_context(x_BK1, mask_BB)
        init_context_BK1 = torch.cat([x_BK1, map_emb_BK.unsqueeze(2)], dim=1)
        context_B1K = self.model.encode_context(init_context_BK1, mask_BB)
        ## for each denoising step (10 steps)
        ## denoise the cur_y_BNK1
        for i in reversed(range(self.cfg.steps)):
            ## i is the time
            cur_y_BNK1 = self.p_sample(x_BK1, mask_BB, cur_y_BNK1, i, context_B1K, map_emb_BK)
        ## final denoised prediction (model predicts residuals wrt anchors)
        prediction_total_BNK1 = cur_y_BNK1
        return prediction_total_BNK1, context_B1K.squeeze(1)
