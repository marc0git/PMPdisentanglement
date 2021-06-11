import sys
from pathlib import Path
from typing import Any, List, Optional, Tuple
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy.spatial import Delaunay
import numpy as np
import gc
import trimesh
import h5py
import os
#os.environ["DISENTANGLEMENT_LIB_DATA"] = "./disentanglement_library/data/"

import random
import pickle
import disentanglement_lib
import disentanglement_lib.data.ground_truth.named_data as named_data 
import disentanglement_lib.data.ground_truth.dsprites 
from torch.utils.data import DataLoader, Dataset

randomstate = np.random.RandomState(100)


def simple_dynamics(z, ground_truth_data, random_state,
                    return_index=False, k=1):
  """Create the pairs."""
  if k == -1:
    k_observed = random_state.randint(1, ground_truth_data.num_factors)
  else:
    k_observed = k
  index_list = random_state.choice(
      z.shape[1], random_state.choice([1, k_observed]), replace=False)
  idx = -1
  for index in index_list:
    z[:, index] = np.random.choice(
        range(ground_truth_data.factors_num_values[index]))
    idx = index
  if return_index:
    return z, idx
  return z, k_observed



class DatasetVariableK(Dataset):
    def __init__(self, dataset_name="shapes3d", factors=None, fraction=1, k=-1):
        if dataset_name=="shapes3d":
            with open('data/shapes3d_byte.pickle', 'rb') as f:
                self.dataset = pickle.load(f)
        else:
            self.dataset = named_data.get_named_ground_truth_data(dataset_name)
#         self.dataset = DATASET
        if factors is None:
            factors = self.dataset.latent_factor_indices
        self.factors = factors
        self.fraction = fraction
        self.k=k
        self.randomstate = np.random.RandomState(100)
    
    def __len__(self) -> int:
        return int(np.prod(self.dataset.factor_sizes)*self.fraction)
#         return int(self.dataset.images.shape[0]*self.fraction)

    def __getitem__(self, item: int):
        ground_truth_data = self.dataset
        sampled_factors = ground_truth_data.sample_factors(1, self.randomstate)
        sampled_observation = ground_truth_data.sample_observations_from_factors(
          sampled_factors, self.randomstate)

        next_factors, index = simple_dynamics(sampled_factors.copy(),
                                            ground_truth_data,
                                            self.randomstate,
                                            return_index=True,
                                            k=self.k)
        next_observation = ground_truth_data.sample_observations_from_factors(next_factors, self.randomstate)
        (np.concatenate((sampled_observation, next_observation), axis=1)[0], [index])

        #handle uint8 data
        scale = 1
        if sampled_observation.dtype==np.uint8:
            scale = 255
            
        return torch.as_tensor(sampled_observation, dtype=torch.float32)[0]/scale,\
               torch.as_tensor(next_observation, dtype=torch.float32)[0]/scale,\
               torch.as_tensor(sampled_factors!=next_factors).int()[0]