import os
import glob
import rasterio
import numpy as np
import pandas as pd
from typing import Optional
import torch
from torch.utils.data import Dataset
from . import transforms

from src.data.challenge import ChallengeData
from src.data.chip import preprocess_chip_data

import warnings

warnings.simplefilter("ignore")


#
# dataset and dataloaders
#
import os
from typing import Optional
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
  def __init__(
    self,
    # images_dir: str,
    provider: ChallengeData,
    ids: Optional[list] = None,
    frac: Optional[float] = None,
    transform_name: Optional[str] = None
  ):
    super().__init__()
    # self.images_dir = images_dir
    self.ids = ids #if ids is not None else os.listdir(images_dir)
    self.provider = provider
    self.transform = transforms.__dict__[transform_name] if transform_name else None

    if frac is not None:
      n = int(len(self.ids) * frac)
      n = n if n > 0 else 1
      self.ids = self.ids[:n]
    
  def __len__(self):
    return len(self.ids)

  def __getitem__(self, i):
    id = self.ids[i]
    chip, agbm = preprocess_chip_data(self.provider, id)
    return dict(
      id = id,
      chip = self.provider.load_image(chip)[0],
      agbm = None if agbm is None else self.provider.load_image(agbm)[0]
    )

class TestSegmentationDataset(SegmentationDataset):
  # def __init__(
  #   self,
  #   images_dir: str,
  #   **kwargs
  # ):
  #   super().__init__(images_dir, **kwargs)

  def __getitem__(self, i):
    item = super().__getitem__(i)
    return {k: item[k] for k in ['id', 'chip']}

# datasets = {
#   'SegmentationDataset': SegmentationDataset,
#   'TestSegmentationDataset': TestSegmentationDataset
# }
