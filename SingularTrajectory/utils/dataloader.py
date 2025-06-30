import os
import math
import json

import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data.dataloader import DataLoader
from .homography import generate_homography, project
from PIL import Image
import torchvision


def get_dataloader(data_dir, phase, obs_len, pred_len, batch_size, skip=1, max_step_size=None, device="cpu"):
    r"""Get dataloader for a specific phase

    Args:
        data_dir (str): path to the dataset directory
        phase (str): phase of the data, one of 'train', 'val', 'test'
        obs_len (int): length of observed trajectory
        pred_len (int): length of predicted trajectory
        batch_size (int): batch size
        max_step_size (float): maximum step size between consecutive observations
            useful to filter out non-realistic trajectories (in training only)

    Returns:
        loader_phase (torch.utils.data.DataLoader): dataloader for the specific phase
    """

    print('Loading', phase, 'data...')

    assert phase in ['train', 'val', 'test']

    pin_memory = False if device == "cpu" else True

    data_set = data_dir + '/' + phase + '/'

    shuffle = True if phase == 'train' else False
    drop_last = True if phase == 'train' else False

    dataset_phase = TrajectoryDataset(data_set, obs_len=obs_len, pred_len=pred_len, skip=skip, max_step_size=max_step_size)
    sampler_phase = None
    if batch_size > 1:
        sampler_phase = TrajBatchSampler(dataset_phase, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    loader_phase = DataLoader(dataset_phase, collate_fn=traj_collate_fn, batch_sampler=sampler_phase, pin_memory=pin_memory)
    return loader_phase


def traj_collate_fn(data):
    r"""Collate function for the dataloader

    Args:
        data (list): list of tuples of (obs_seq, pred_seq, non_linear_ped, loss_mask, seq_start_end, scene_id)

    Returns:
        obs_seq_list (torch.Tensor): (num_ped, obs_len, 2)
        pred_seq_list (torch.Tensor): (num_ped, pred_len, 2)
        non_linear_ped_list (torch.Tensor): (num_ped,)
        loss_mask_list (torch.Tensor): (num_ped, obs_len + pred_len)
        scene_mask (torch.Tensor): (num_ped, num_ped)
        seq_start_end (torch.Tensor): (num_ped, 2)
        scene_id
    """

    data_collated = {}
    for k in data[0].keys():
        data_collated[k] = [d[k] for d in data]

    _len = [len(seq) for seq in data_collated["obs_traj"]]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end] for start, end in zip(cum_start_idx, cum_start_idx[1:])]
    seq_start_end = torch.LongTensor(seq_start_end)
    scene_mask = torch.zeros(sum(_len), sum(_len), dtype=torch.bool)
    for idx, (start, end) in enumerate(seq_start_end):
        scene_mask[start:end, start:end] = 1

    data_collated["obs_traj"] = torch.cat(data_collated["obs_traj"], dim=0)
    data_collated["pred_traj"] = torch.cat(data_collated["pred_traj"], dim=0)
    data_collated["anchor"] = torch.cat(data_collated["anchor"], dim=0)
    data_collated["non_linear_ped"] = torch.cat(data_collated["non_linear_ped"], dim=0)
    data_collated["loss_mask"] = torch.cat(data_collated["loss_mask"], dim=0)
    data_collated["scene_mask"] = scene_mask
    data_collated["seq_start_end"] = seq_start_end
    data_collated["frame"] = torch.cat(data_collated["frame"], dim=0)
    data_collated["scene_id"] = np.concatenate(data_collated["scene_id"], axis=0)

    if data[0]["dataset_name"] == "pfsd":
        data_collated["scene_mask"] = torch.zeros(sum(_len), sum(_len), dtype=torch.bool)

    return data_collated


class TrajBatchSampler(Sampler):
    r"""Samples batched elements by yielding a mini-batch of indices.

    =
    Tries to generate mini-batches with approximately equal number of
    pedestrians in each batch. Incude a number of sequences in each batch
    such that the total number of pedestrians in each batch is equal to
    or greater than the batch size.
    =

    Args:
        data_source (Dataset): dataset to sample from
        batch_size (int): Size of mini-batch.
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``
        generator (Generator): Generator used in sampling.
    """

    def __init__(self, data_source, batch_size=64, shuffle=False, drop_last=False, generator=None):
        self.data_source = data_source
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.generator = generator

    def __iter__(self):
        assert len(self.data_source) == len(self.data_source.num_peds_in_seq)

        if self.shuffle:
            if self.generator is None:
                generator = torch.Generator()
                generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            else:
                generator = self.generator
            indices = torch.randperm(len(self.data_source), generator=generator).tolist()
        else:
            indices = list(range(len(self.data_source)))
        num_peds_indices = self.data_source.num_peds_in_seq[indices]

        batch = []
        total_num_peds = 0
        for idx, num_peds in zip(indices, num_peds_indices):
            batch.append(idx)
            total_num_peds += num_peds
            if total_num_peds >= self.batch_size:
                yield batch
                batch = []
                total_num_peds = 0
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        # Approximated number of batches.
        # The order of trajectories can be shuffled, so this number can vary from run to run.
        if self.drop_last:
            return sum(self.data_source.num_peds_in_seq) // self.batch_size
        else:
            return (sum(self.data_source.num_peds_in_seq) + self.batch_size - 1) // self.batch_size


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold):
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non-linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0

def _load_homographies(scene_name: str,
                       homographies_folder: str) -> tuple[torch.Tensor,
                                                          torch.Tensor,
                                                          torch.Tensor,
                                                          torch.Tensor,
                                                          torch.Tensor]:
    """Retrieves the homography matrix (json file).

    Args:
        scene_name: The name of the scene.
        dataset_folder: The folder containing the dataset.
            Expected to contain the 'meters' and 'mask' folders,
            each containing the homography matrices for the scene
            in JSON format.

    Returns:
        Tuple containing 3 homography matrices:
        - homography_orig2meters: The homography matrix from the original
            coordinate system to the meters coordinate system.
        - homography_meters2orig: The homography matrix from the meters
            coordinate system to the original coordinate system.
        - homography_meters2mask: The homography matrix from the meters
            coordinate system to the mask coordinate system.
    """

    homography_orig2meters_path = \
        os.path.join(homographies_folder, 'meters', f'{scene_name}.json')

    # Load the homography_orig2meters from the JSON file.
    with open(homography_orig2meters_path, 'r') as f:
        homography_orig2meters = torch.tensor(json.load(f))

    # Compute the inverse of homography_orig2meters.
    homography_meters2orig = torch.inverse(homography_orig2meters)

    # Load the homography_meters2mask from the JSON file.
    homography_meters2mask_path = \
        os.path.join(homographies_folder, 'mask', f'{scene_name}.json')
    with open(homography_meters2mask_path, 'r') as f:
        homography_meters2mask = torch.tensor(json.load(f))

    # Load the homography_image2meters from the JSON file.
    homography_image2meters_path = \
        os.path.join(homographies_folder, 'image', f'{scene_name}.json')
    with open(homography_image2meters_path, 'r') as f:
        homography_image2meters = torch.tensor(json.load(f))

    # Compute the inverse of homography_image2meters.
    homography_meters2image = torch.inverse(homography_image2meters)


    return (homography_orig2meters.float(),
            homography_meters2orig.float(),
            homography_meters2mask.float(),
            homography_image2meters.float(),
            homography_meters2image.float())


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.02, min_ped=1, max_step_size=None, delim='\t'):
        """
        Args:
        - data_dir: Directory containing dataset files in the format <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non-linear traj when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a sequence
        - max_step_size: Maximum step size between consecutive observations,
            useful to filter out non-realistic trajectories (in training only)
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()

        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = sorted(os.listdir(self.data_dir))
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files if _path.endswith('.txt')]

        num_peds_in_seq = []
        seq_list = []
        loss_mask_list = []
        non_linear_ped = []
        frame_list = []
        scene_id = []
        self.homography = {}
        self.vector_field = {}
        self.maps = {}
        scene_img_map = {'biwi_eth': 'seq_eth', 'biwi_hotel': 'seq_hotel',
                         'students001': 'students003', 'students003': 'students003', 'uni_examples': 'students003',
                         'crowds_zara01': 'crowds_zara01', 'crowds_zara02': 'crowds_zara02', 'crowds_zara03': 'crowds_zara02'}

        for path in all_files:
            parent_dir, scene_name = os.path.split(path)
            parent_dir, phase = os.path.split(parent_dir)
            parent_dir, dataset_name = os.path.split(parent_dir)
            scene_name, _ = os.path.splitext(scene_name)
            scene_name = scene_name.replace('_' + phase, '') # eth/ucy
            scene_name = scene_name.replace(phase + '_', '') # sdd

            self.dataset_name = dataset_name

            if dataset_name in ["eth", "hotel", "univ", "zara1", "zara2"]:
                if scene_name not in self.homography:
                    homography_folder = os.path.join(parent_dir, "homography", "eth_ucy")
                    hom_orig2meters, hom_meters2orig, hom_meters2mask, hom_image2meters, hom_meters2image = \
                        _load_homographies(scene_name, homography_folder)
                    self.homography[scene_name] = {
                        "orig2meters": hom_orig2meters.numpy(),
                        "meters2orig": hom_meters2orig.numpy(),
                        "meters2mask": hom_meters2mask.numpy(),
                        "meters2image": hom_meters2image.numpy(),
                        "image2meters": hom_image2meters.numpy(),
                    }

                self.vector_field[scene_name] = np.load(os.path.join(parent_dir, "vectorfield", scene_img_map[scene_name] + "_vector_field.npy"))

                img_path = os.path.join(parent_dir, "image", scene_name + "-mask.png")
                self.maps[scene_name] = torchvision.io.read_image(img_path) / 255.0

            elif dataset_name in ("sdd", "pfsd", "thor"):
                if scene_name not in self.homography:
                    homography_folder = os.path.join(parent_dir, "homography", dataset_name)
                    hom_orig2meters, hom_meters2orig, hom_meters2mask, hom_image2meters, hom_meters2image = \
                        _load_homographies(scene_name, homography_folder)
                    self.homography[scene_name] = {
                        "orig2meters": hom_orig2meters.numpy(),
                        "meters2orig": hom_meters2orig.numpy(),
                        "meters2mask": hom_meters2mask.numpy(),
                        "meters2image": hom_meters2image.numpy(),
                        "image2meters": hom_image2meters.numpy(),
                    }

                self.vector_field[scene_name] = np.load(os.path.join(parent_dir, "vectorfield", scene_name + "_vector_field.npy"))

                img_path = os.path.join(parent_dir, "image", scene_name + "-mask.png")
                self.maps[scene_name] = torchvision.io.read_image(img_path) / 255.0

            elif dataset_name in [aa + '2' + bb for aa in ['A', 'B', 'C', 'D', 'E'] for bb in ['A', 'B', 'C', 'D', 'E'] if aa != bb]:
                homography_file = os.path.join(parent_dir, "homography", scene_name + "_H.txt")
                self.homography[scene_name] = np.loadtxt(homography_file)

            # Load data
            data = read_file(path, delim)
            if data.shape[0] == 0:
                ## Skip the scene if no data is present.
                continue

            frames = np.unique(data[:, 0]).tolist()
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :])
            num_sequences = int(math.ceil((len(frames) - self.seq_len + 1) / skip))

            ## frame_data[i] has shape (num_peds_in_frame_i, 4)

            for idx in range(0, num_sequences * self.skip + 1, skip):
                ## Data for the current sequence of seq_len (=20) frames.
                ## B = sum(num_peds_in_frame_i for i in [idx, idx + seq_len])
                ## 4 = (frame_id, ped_id, x, y)
                curr_seq_data_B4 = np.concatenate(frame_data[idx:idx + self.seq_len], axis=0)
                ## Pedestrian IDs in the current sequence.
                peds_in_curr_seq_S = np.unique(curr_seq_data_B4[:, 1])
                ## Empty scene array.
                curr_seq_S2T = np.zeros((len(peds_in_curr_seq_S), 2, self.seq_len))
                ## Empty mask array: likely used to mask "visible" pedestrians at each time-step.
                curr_loss_mask_ST = np.zeros((len(peds_in_curr_seq_S), self.seq_len))
                num_peds_considered = 0
                _non_linear_ped = []
                ## For each pedestrian in the current sequence.
                for _, ped_id in enumerate(peds_in_curr_seq_S):
                    ## Data for the current pedestrian.
                    # Shape: (num_frames_ped_i, 4)
                    curr_ped_seq_T4 = curr_seq_data_B4[curr_seq_data_B4[:, 1] == ped_id, :]
                    curr_ped_seq_T4 = np.around(curr_ped_seq_T4, decimals=4)
                    ## curr_ped_seq_T4[0, 0] is the frame_id of the first
                    ## appearance of the current pedestrian.

                    ## Checks if the current pedestrian is visible in all the frames of the current sequence.
                    pad_front = frames.index(curr_ped_seq_T4[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq_T4[-1, 0]) - idx + 1

                    if (pad_end - pad_front != self.seq_len) or (curr_ped_seq_T4.shape[0] != self.seq_len):
                        continue

                    curr_ped_seq_2T = np.transpose(curr_ped_seq_T4[:, 2:])
                    # curr_ped_seq_2T = curr_ped_seq_2T

                    ## Check if the step size between consecutive observations is realistic.

                    if (phase != 'test'
                        and dataset_name == "sdd"
                        and max_step_size is not None
                        and np.linalg.norm(np.diff(curr_ped_seq_2T, axis=1), axis=0).max() > max_step_size):
                        continue

                    _idx = num_peds_considered
                    curr_seq_S2T[_idx, :, pad_front:pad_end] = curr_ped_seq_2T
                    # Linear vs Non-Linear Trajectory
                    _non_linear_ped.append(poly_fit(curr_ped_seq_2T, pred_len, threshold))
                    curr_loss_mask_ST[_idx, pad_front:pad_end] = 1
                    num_peds_considered += 1

                ## Make sure the scene has at least min_ped pedestrians.
                if num_peds_considered > min_ped:
                    non_linear_ped += _non_linear_ped
                    ## Number of pedestrians in the current scene.
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask_ST[:num_peds_considered])
                    ## Append to the actual list of scenes.
                    seq_list.append(curr_seq_S2T[:num_peds_considered])
                    frame_list.extend([frames[idx]] * num_peds_considered)
                    scene_id.extend([scene_name] * num_peds_considered)

        self.num_seq = len(seq_list)
        seq_list_B2T = np.concatenate(seq_list, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        non_linear_ped = np.asarray(non_linear_ped)
        self.num_peds_in_seq = np.array(num_peds_in_seq)
        self.frame_list = np.array(frame_list, dtype=np.int32)
        self.scene_id = np.array(scene_id)

        # Convert numpy -> Torch Tensor

        ## This is a vector of all the observed trajectories.
        ## To access a particular scene, you need to access the tensors
        ## in the range contained in seq_start_end.
        self.obs_traj_BT2 = torch.from_numpy(seq_list_B2T[:, :, :self.obs_len]).type(torch.float).permute(0, 2, 1)  # NTC
        self.pred_traj_BT2 = torch.from_numpy(seq_list_B2T[:, :, self.obs_len:]).type(torch.float).permute(0, 2, 1)  # NTC

        ## project trajectories to meters.
        print("projecting trajectories to meters")
        obs_traj_meters_list = []
        pred_traj_meters_list = []
        for obs_traj_T2, pred_traj_T2, scene_id in zip(self.obs_traj_BT2, self.pred_traj_BT2, self.scene_id):
            hom_orig2meters = torch.from_numpy(self.homography[scene_id]["orig2meters"]).type(torch.float)
            obs_traj_T2 = project(obs_traj_T2, hom_orig2meters)
            pred_traj_T2 = project(pred_traj_T2, hom_orig2meters)
            obs_traj_meters_list.append(obs_traj_T2)
            pred_traj_meters_list.append(pred_traj_T2)
        self.obs_traj_BT2 = torch.stack(obs_traj_meters_list, dim=0)
        self.pred_traj_BT2 = torch.stack(pred_traj_meters_list, dim=0)
        print("done projecting trajectories to meters")

        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float).gt(0.5)
        self.non_linear_ped = torch.from_numpy(non_linear_ped).type(torch.float).gt(0.5)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [(start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])]
        self.frame_list = torch.from_numpy(self.frame_list).type(torch.long)
        self.anchor = None


    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = {"obs_traj": self.obs_traj_BT2[start:end],
               "pred_traj": self.pred_traj_BT2[start:end],
               "anchor": self.anchor[start:end],
               "non_linear_ped": self.non_linear_ped[start:end],
               "loss_mask": self.loss_mask[start:end],
               "scene_mask": None,
               "seq_start_end": [[0, end - start]],
               "frame": self.frame_list[start:end],
               "scene_id": self.scene_id[start:end],
               "dataset_name": self.dataset_name}
        return out
