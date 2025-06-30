import os
import pickle
import torch
import numpy as np
from tqdm import tqdm
from . import *

import utils.homography as hm
import baseline.transformerdiffusion.model_utils as model_utils

class STTrainer:
    r"""Base class for all Trainers"""

    def __init__(self, args, hyper_params, device):
        print("Trainer initiating...")

        # Reproducibility
        reproducibility_settings(seed=0)

        self.args, self.hyper_params = args, hyper_params
        self.model, self.optimizer, self.scheduler = None, None, None
        self.loader_train, self.loader_val, self.loader_test = None, None, None
        self.dataset_dir = hyper_params.dataset_dir + hyper_params.dataset + '/'
        self.checkpoint_dir = hyper_params.checkpoint_dir + '/' + args.tag + '/' + hyper_params.dataset + '/'
        print("Checkpoint dir:", self.checkpoint_dir)
        self.log = {'train_loss': [], 'val_loss': []}
        self.stats_func, self.stats_meter = None, None
        self.reset_metric()

        device = 'cpu'
        if args.device == 'gpu':
            if torch.cuda.is_available():
                if args.gpu_id is not None:
                    device = f'cuda:{args.gpu_id}'
                else:
                    device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'
        self.device = device

        if not args.test:
            # Save arguments and configs
            if not os.path.exists(self.checkpoint_dir):
                os.makedirs(self.checkpoint_dir)

            with open(self.checkpoint_dir + 'args.pkl', 'wb') as fp:
                pickle.dump(args, fp)

            with open(self.checkpoint_dir + 'config.pkl', 'wb') as fp:
                pickle.dump(hyper_params, fp)

    def init_descriptor(self):
        # Singular space initialization

        ## Training only

        print("Singular space initialization...")
        obs_traj_BT2, pred_traj_BT2 = self.loader_train.dataset.obs_traj_BT2, self.loader_train.dataset.pred_traj_BT2
        obs_traj_BT2 = obs_traj_BT2.to(self.device)
        pred_traj_BT2 = pred_traj_BT2.to(self.device)
        obs_traj_BT2, pred_traj_BT2 = augment_trajectory(obs_traj_BT2, pred_traj_BT2)
        self.model.calculate_parameters(obs_traj_BT2, pred_traj_BT2)
        print("Anchor generation...")

    def init_adaptive_anchor(self, dataset):
        ## run at every epoch, so can use the learned V_trunc, ...

        print("Adaptive anchor initialization...")
        ## ask to the model to compute the adaptive anchor
        dataset.anchor = self.model.calculate_adaptive_anchor(dataset)

    def train(self, epoch):
        raise NotImplementedError

    @torch.no_grad()
    def valid(self, epoch):
        raise NotImplementedError

    @torch.no_grad()
    def test(self):
        raise NotImplementedError

    def fit(self):
        print("Training started...")
        for epoch in range(self.hyper_params.num_epochs):
            self.train(epoch)
            self.valid(epoch)

            if self.hyper_params.lr_schd:
                self.scheduler.step()

            # Save the best model
            if epoch == 0 or self.log['val_loss'][-1] < min(self.log['val_loss'][:-1]):
                self.save_model()

            print(" ")
            print("Dataset: {0}, Epoch: {1}".format(self.hyper_params.dataset, epoch))
            print("Train_loss: {0:.8f}, Val_los: {1:.8f}".format(self.log['train_loss'][-1], self.log['val_loss'][-1]))
            print("Min_val_epoch: {0}, Min_val_loss: {1:.8f}".format(np.array(self.log['val_loss']).argmin(),
                                                                     np.array(self.log['val_loss']).min()))
            print(" ")
        print("Done.")

    def reset_metric(self):
        self.stats_func = {'ADE': compute_batch_ade, 'FDE': compute_batch_fde, 'ENV_COL': None}
        self.stats_meter = {x: AverageMeter() for x in self.stats_func.keys()}

    def get_metric(self):
        return self.stats_meter

    def load_model(self, filename='model_best.pth'):
        model_path = self.checkpoint_dir + filename
        self.model.load_state_dict(torch.load(model_path))

    def save_model(self, filename='model_best.pth'):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        model_path = self.checkpoint_dir + filename
        torch.save(self.model.state_dict(), model_path)


class STSequencedMiniBatchTrainer(STTrainer):
    r"""Base class using sequenced mini-batch training strategy"""

    def __init__(self, args, hyper_params, device):
        super().__init__(args, hyper_params, device)

        # Dataset preprocessing
        obs_len, pred_len, skip = hyper_params.obs_len, hyper_params.pred_len, hyper_params.skip
        max_step_size = hyper_params.max_step_size
        self.loader_train = get_dataloader(self.dataset_dir, 'train', obs_len, pred_len, batch_size=1, skip=skip, max_step_size=max_step_size, device=self.device)
        self.loader_val = get_dataloader(self.dataset_dir, 'val', obs_len, pred_len, batch_size=1, device=self.device)
        self.loader_test = get_dataloader(self.dataset_dir, 'test', obs_len, pred_len, batch_size=1, device=self.device)

    def train(self, epoch):
        self.model.train()
        loss_batch = 0
        is_first_loss = True
        device = self.device

        for cnt, batch in enumerate(tqdm(self.loader_train, desc=f'Train Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = [tensor.to(device=device, non_blocking=True) for tensor in batch[:2]]

            self.optimizer.zero_grad()

            output = self.model(obs_traj, pred_traj)

            loss = output["loss_euclidean_ade"]
            loss[torch.isnan(loss)] = 0

            if (cnt + 1) % self.hyper_params.batch_size != 0 and (cnt + 1) != len(self.loader_train):
                if is_first_loss:
                    is_first_loss = False
                    loss_cum = loss
                else:
                    loss_cum += loss

            else:
                is_first_loss = True
                loss_cum += loss
                loss_cum /= self.hyper_params.batch_size
                loss_cum.backward()

                if self.hyper_params.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_params.clip_grad)

                self.optimizer.step()
                loss_batch += loss_cum.item()

        self.log['train_loss'].append(loss_batch / len(self.loader_train))

    @torch.no_grad()
    def valid(self, epoch):
        self.model.eval()
        loss_batch = 0
        device = self.device

        for cnt, batch in enumerate(tqdm(self.loader_val, desc=f'Valid Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = [tensor.to(device=device, non_blocking=True) for tensor in batch[:2]]

            output = self.model(obs_traj, pred_traj)

            recon_loss = output["loss_euclidean_fde"] * obs_traj.size(0)
            loss_batch += recon_loss.item()

        num_ped = sum(self.loader_val.dataset.num_peds_in_seq)
        self.log['val_loss'].append(loss_batch / num_ped)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.reset_metric()
        device = self.device

        for batch in tqdm(self.loader_test, desc=f"Test {self.hyper_params.dataset.upper()} scene"):
            obs_traj, pred_traj = [tensor.to(device=device, non_blocking=True) for tensor in batch[:2]]

            output = self.model(obs_traj)

            # Evaluate trajectories
            for metric in self.stats_func.keys():
                value = self.stats_func[metric](output["recon_traj"], pred_traj)
                self.stats_meter[metric].extend(value)

        return {x: self.stats_meter[x].mean() for x in self.stats_meter.keys()}


class STCollatedMiniBatchTrainer(STTrainer):
    r"""Base class using collated mini-batch training strategy"""

    def __init__(self, args, hyper_params, device):
        super().__init__(args, hyper_params, device)

        # Dataset preprocessing
        batch_size = hyper_params.batch_size
        obs_len, pred_len, skip = hyper_params.obs_len, hyper_params.pred_len, hyper_params.skip
        max_step_size = hyper_params.max_step_size
        self.loader_train = get_dataloader(self.dataset_dir, 'train', obs_len, pred_len, batch_size=batch_size, skip=skip, max_step_size=max_step_size, device=self.device)
        self.loader_val = get_dataloader(self.dataset_dir, 'val', obs_len, pred_len, batch_size=batch_size, device=self.device)
        self.loader_test = get_dataloader(self.dataset_dir, 'test', obs_len, pred_len, batch_size=1, device=self.device)

    def train(self, epoch):
        self.model.train()
        loss_batch = 0
        device = self.device

        for cnt, batch in enumerate(tqdm(self.loader_train, desc=f'Train Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = [tensor.to(device=device, non_blocking=True) for tensor in batch[:2]]

            self.optimizer.zero_grad()

            output = self.model(obs_traj, pred_traj)

            loss = output["loss_euclidean_ade"]
            loss[torch.isnan(loss)] = 0
            loss_batch += loss.item()

            loss.backward()
            if self.hyper_params.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_params.clip_grad)
            self.optimizer.step()

        self.log['train_loss'].append(loss_batch / len(self.loader_train))

    @torch.no_grad()
    def valid(self, epoch):
        self.model.eval()
        loss_batch = 0
        device = self.device

        for cnt, batch in enumerate(tqdm(self.loader_val, desc=f'Valid Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = [tensor.to(device=device, non_blocking=True) for tensor in batch[:2]]

            output = self.model(obs_traj, pred_traj)

            recon_loss = output["loss_euclidean_fde"] * obs_traj.size(0)
            loss_batch += recon_loss.item()

        num_ped = sum(self.loader_val.dataset.num_peds_in_seq)
        self.log['val_loss'].append(loss_batch / num_ped)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.reset_metric()
        device = self.device

        for batch in tqdm(self.loader_test, desc=f"Test {self.hyper_params.dataset.upper()} scene"):
            obs_traj, pred_traj = [tensor.to(device=device, non_blocking=True) for tensor in batch[:2]]

            output = self.model(obs_traj)

            # Evaluate trajectories
            for metric in self.stats_func.keys():
                value = self.stats_func[metric](output["recon_traj"], pred_traj)
                self.stats_meter[metric].extend(value)

        return {x: self.stats_meter[x].mean() for x in self.stats_meter.keys()}


class STTransformerDiffusionTrainer(STCollatedMiniBatchTrainer):
    r"""SingularTrajectory model trainer"""

    def __init__(self, base_model, model, hook_func, args, hyper_params, device):
        super().__init__(args, hyper_params, device)
        ## diffusion params
        cfg = DotDict({'scheduler': 'ddim', 'steps': 10, 'beta_start': 1.e-4, 'beta_end': 5.e-2, 'beta_schedule': 'linear',
                       'k': hyper_params.k, 's': hyper_params.num_samples, 'baseline_use_map': hyper_params.baseline_use_map,})

        device = 'cpu'
        if args.device == 'gpu':
            if torch.cuda.is_available():
                if args.gpu_id is not None:
                    device = f'cuda:{args.gpu_id}'
                else:
                    device = 'cuda'
            elif torch.backends.mps.is_available():
                device = 'mps'

        ## diffusion model
        predictor_model = base_model(cfg).to(device)
        ## SingularTrajectory model
        eigentraj_model = model(baseline_model=predictor_model, hook_func=hook_func, hyper_params=hyper_params, device=device).to(device)
        self.model = eigentraj_model
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=hyper_params.lr,
                                           weight_decay=hyper_params.weight_decay)

        ## learning rate scheduler
        if hyper_params.lr_schd:
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer=self.optimizer,
                                                             step_size=hyper_params.lr_schd_step,
                                                             gamma=hyper_params.lr_schd_gamma)

## shapes
## B: batch size
## K: embedding space size
## N: number of samples

    def train(self, epoch):
        self.model.train()
        loss_batch = 0
        device = self.device

        if self.loader_train.dataset.anchor is None:
            self.init_adaptive_anchor(self.loader_train.dataset)

        # maps to correct device
        maps = self.loader_train.dataset.maps
        for key in maps.keys():
            maps[key] = maps[key].to(device=device, non_blocking=True)

        for cnt, batch in enumerate(tqdm(self.loader_train, desc=f'Train Epoch {epoch}', mininterval=1)):
            ## get past observed traj and future gt traj
            obs_traj_BT2, pred_traj_BT2 = batch["obs_traj"].to(device=device, non_blocking=True), batch["pred_traj"].to(device=device, non_blocking=True)
            ## get adaptive anchor
            adaptive_anchor_BKN = batch["anchor"].to(device=device, non_blocking=True)
            ## get scene mask (and sequence start/end (not used))
            scene_mask_BB, seq_start_end = batch["scene_mask"].to(device=device, non_blocking=True), batch["seq_start_end"].to(device=device, non_blocking=True)

            self.optimizer.zero_grad()

            additional_information = {
                "scene_mask": scene_mask_BB,
                "num_samples": self.hyper_params.num_samples,
                "maps": maps,
                "homography": self.loader_train.dataset.homography,
                "vector_field": self.loader_train.dataset.vector_field,
                "scene_ids": batch["scene_id"],
                "epoch": epoch,
            }
            # print("obs_traj_BT2", obs_traj_BT2.shape)
            # print("pred_traj_BT2", pred_traj_BT2.shape)
            output = self.model(obs_traj_BT2, adaptive_anchor_BKN, pred_traj_BT2, addl_info=additional_information)
            # print("output", output["recon_traj"].shape)

            ## get the loss (that is already computed in the model forward pass)
            loss_main = output["loss_euclidean_ade"]
            ## set nan values of the loss to 0
            loss_main[torch.isnan(loss_main)] = 0


            map_nce_loss = output["loss_map_nce"] * self.hyper_params.map_nce_weight
            # map_nce_loss = output["loss_map_nce"] * 0

            if epoch < self.hyper_params.env_col_loss_epoch:
                env_col_loss = torch.tensor(0.0, device=device)
            else:
                env_col_loss = output["loss_env_collision"] * self.hyper_params.env_col_weight
            # env_col_loss = output["loss_env_collision"] * 0

            diversity_loss = output["loss_diversity"] * self.hyper_params.diversity_loss_weight

            ## accumulate the loss for all the batches

            loss = loss_main + map_nce_loss + env_col_loss + diversity_loss

            loss_batch += loss.item()

            # print(f"Loss: {loss.item()}, Loss_main: {loss_main.item()}, Loss_map_nce: {map_nce_loss.item()}, Loss_env_col: {env_col_loss.item()}, Loss_diversity: {diversity_loss.item()}")


            loss.backward()
            if self.hyper_params.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hyper_params.clip_grad)
            self.optimizer.step()

        ## average loss of the epoch
        self.log['train_loss'].append(loss_batch / len(self.loader_train))

    @torch.no_grad()
    def valid(self, epoch):
        self.model.eval()
        loss_batch = 0
        device = self.device

        if self.loader_val.dataset.anchor is None:
            self.init_adaptive_anchor(self.loader_val.dataset)

        # maps to correct device
        maps = self.loader_val.dataset.maps
        for key in maps.keys():
            maps[key] = maps[key].to(device=device, non_blocking=True)

        for cnt, batch in enumerate(tqdm(self.loader_val, desc=f'Valid Epoch {epoch}', mininterval=1)):
            obs_traj, pred_traj = batch["obs_traj"].to(device=device, non_blocking=True), batch["pred_traj"].to(device=device, non_blocking=True)
            adaptive_anchor = batch["anchor"].to(device=device,  non_blocking=True)
            scene_mask, seq_start_end = batch["scene_mask"].to(device=device, non_blocking=True), batch["seq_start_end"].to(device=device, non_blocking=True)

            additional_information = {
                "scene_mask": scene_mask,
                "num_samples": self.hyper_params.num_samples,
                "maps": maps,
                "homography": self.loader_val.dataset.homography,
                "vector_field": self.loader_val.dataset.vector_field,
                "scene_ids": batch["scene_id"],
                "epoch": epoch,
            }
            output = self.model(obs_traj, adaptive_anchor, pred_traj, addl_info=additional_information)

            recon_loss = output["loss_euclidean_fde"] * obs_traj.size(0)
            loss_batch += recon_loss.item()

        num_ped = sum(self.loader_val.dataset.num_peds_in_seq)
        self.log['val_loss'].append(loss_batch / num_ped)

    @torch.no_grad()
    def test(self):
        self.model.eval()
        self.reset_metric()

        if self.loader_test.dataset.anchor is None:
            self.init_adaptive_anchor(self.loader_test.dataset)

        # Dataset name.
        dataset_name = self.loader_test.dataset.data_dir.split('/')[2]

        print("=====================================")
        print(dataset_name)
        print("=====================================")

        # Create a list to store the results.
        results = []
        obs_traj_list = []
        pred_traj_list = []
        scene_name_list = []
        device = self.device

        # maps to correct device
        maps = self.loader_test.dataset.maps
        for key in maps.keys():
            maps[key] = maps[key].to(device=device, non_blocking=True)

        for cnt, batch in enumerate(tqdm(self.loader_test, desc=f"Test {self.hyper_params.dataset.upper()} scene")):
            ## NOTE: the batch size is 1, so a single scene is processed at a time
            obs_traj, pred_traj = batch["obs_traj"].to(device=device, non_blocking=True), batch["pred_traj"].to(device=device, non_blocking=True)
            adaptive_anchor = batch["anchor"].to(device=device, non_blocking=True)
            scene_mask, seq_start_end = batch["scene_mask"].to(device=device, non_blocking=True), batch["seq_start_end"].to(device=device, non_blocking=True)

            additional_information = {
                "scene_mask": scene_mask,
                "num_samples": self.hyper_params.num_samples,
                "maps": maps,
                "homography": self.loader_test.dataset.homography,
                "vector_field": self.loader_test.dataset.vector_field,
                "scene_ids": batch["scene_id"],
                "epoch": 0,
            }
            # output["recon_traj"].shape: [N, B, P, 2]
            output = self.model(obs_traj, adaptive_anchor, addl_info=additional_information)
            recon_traj_meters_NBP2 = output["recon_traj"]

            scene_name = batch["scene_id"][0]
            scene_name_list.append(scene_name)

            ## transform to original space
            hom_meters2orig = self.loader_test.dataset.homography[scene_name]["meters2orig"]
            hom_meters2orig = torch.from_numpy(hom_meters2orig).to(device=device)
            recon_traj_orig_NBP2 = hm.project(recon_traj_meters_NBP2, hom_meters2orig)
            obs_traj_orig_BO2 = hm.project(obs_traj, hom_meters2orig)
            pred_traj_orig_BP2 = hm.project(pred_traj, hom_meters2orig)

            # Compute ENV_COL metric.
            hom_meters2mask = self.loader_test.dataset.homography[scene_name]["meters2mask"]
            hom_meters2mask = torch.from_numpy(hom_meters2mask).to(device=device)
            recon_traj_meters_AP2 = recon_traj_meters_NBP2.reshape(-1, 12, 2)
            env_collisions_A = model_utils.check_env_collisions(
                recon_traj_meters_AP2,
                self.loader_test.dataset.maps[scene_name],
                scene_transform_matrix=torch.eye(3, device=device),
                homography_meters2mask=hom_meters2mask,
            )
            env_collisions_NB = env_collisions_A.reshape(self.hyper_params.num_samples, -1)
            env_col_list = [
                env_collisions_NB[:, i].sum() / env_collisions_NB.size(0) * 100
                for i in range(env_collisions_NB.size(1))
            ]

            # Add the results to the list.
            results.append(recon_traj_orig_NBP2.permute(1, 0, 2, 3))
            obs_traj_list.append(obs_traj_orig_BO2)
            pred_traj_list.append(pred_traj_orig_BP2)

            for metric in self.stats_func.keys():
                if metric == "ENV_COL":
                    self.stats_meter[metric].extend(env_col_list)
                else:
                    value = self.stats_func[metric](recon_traj_orig_NBP2, pred_traj_orig_BP2)
                    self.stats_meter[metric].extend(value)

        # Save the results.
        # os.makedirs("results", exist_ok=True)
        os.makedirs(f"results/{dataset_name}", exist_ok=True)
        torch.save(results, f"results/{dataset_name}/traj_hat.pt")
        torch.save(obs_traj_list, f"results/{dataset_name}/traj_obs.pt")
        torch.save(pred_traj_list, f"results/{dataset_name}/traj_pred.pt")
        torch.save(scene_name_list, f"results/{dataset_name}/scene_names.pt")

        return {x: self.stats_meter[x].mean() for x in self.stats_meter.keys()}
