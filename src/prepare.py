import sys
import torch
import torchvision

import addict
import argparse

from .data import ChallengeData, ConfigManager

def test_tiff_profile(filename):
  import rasterio
  with rasterio.open(filename) as fobj:
    print(fobj.profile)

def processe_input_chips(challenge, folds=None, nworkers=4, batch_size=36):
  from tqdm import tqdm
  from concurrent.futures import ThreadPoolExecutor
  from concurrent.futures import as_completed
  from .data.chip import preprocess_chip_data

  if folds is None:
    folds = list(range(challenge.nfolds))
  elif not isinstance(folds, list):
    folds = [folds]
  
  chip_ids = []
  for fold in folds:
    chip_ids.extend(challenge.get_ids(fold))

  batches = []
  batch_size = batch_size if batch_size > 0 else 1
  nbatches = (len(chip_ids) + batch_size-1) // batch_size
  for i in range(nbatches):
    batches.append(chip_ids[i*batch_size: (i+1)*batch_size])

  for batch in tqdm(batches):
    with ThreadPoolExecutor(nworkers) as executor:
      futures = [executor.submit(preprocess_chip_data, challenge, chip_id) for chip_id in batch]
      for future in as_completed(futures):
        pass

def main(args):

  S1_NODATA = -9999
  S2_CLP_NODATA = 255
  IMAGE_STATS_FILENAME = 'image_stats.csv'

  challenge_config = addict.Dict({
    's3region': 'as',
    's3bucket': 'drivendata-competition-biomassters-public',
    'metadata': {
      'features': 'features_metadata.csv',
      'train_agbm': 'train_agbm_metadata.csv',
      'features_stats': IMAGE_STATS_FILENAME,
    },
    'directories': {
      'train': 'train_features',
      'test': 'test_features',
      'train_agbm': 'train_agbm'
    },
    'nodata': {
      's1': [S1_NODATA for _ in range(4)],
      's2': [None for _ in range(10)] + [S2_CLP_NODATA]
    },
    'id_str': 'chip_id',
    'nfolds': 5,
    'base_dir': 'data',
  })

  challenge = ChallengeData(challenge_config)
  stats = challenge.get_stats()

  config_manager = ConfigManager('configs')
  config_manager.generate(challenge_config)

  # processe_input_chips(challenge, folds=-1)

  # print(config_manager.next())
  # cfg = addict.Dict(config_manager.parse(config=config_manager.next()))
  # print(cfg.data.stats)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # parser.add_argument(
  #   '--configs', 
  # )

  args = parser.parse_args()
  main(args)
  sys.exit(0)