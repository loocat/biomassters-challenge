import rasterio
import warnings
import numpy as np

from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed

#
# helper functions
#
def post_transform(image, **kwargs):
  if image.ndim == 3:
    return image.transpose(2, 0, 1).astype('float32')
  else:
    return image.astype('float32')

def save_raster(path, raster, **profile):
  """Save raster on disk"""
  c, h, w = raster.shape
  _profile = dict(
    driver="GTiff",
    height=h,
    width=w,
    count=c,
    dtype=raster.dtype,
  )
  _profile.update(profile)

  with rasterio.open(path, "w", **_profile) as dst:
    dst.write(raster)
  # tiffile.imwrite(path, raster.transpose(1, 2, 0), **_profile)

def load_chip_data(challenge, s3paths, nworkers=4):
  chip_data = {
    's1': {},
    's2': {},
    'agbm': []
  }

  def assign(result):
    image, s3path = result

    splits = Path(s3path).stem.split('_')
    month = splits.pop()
    sentinel = splits.pop().lower()

    nodata = (image == int(challenge.nodata[sentinel][-1])).sum()

    chip_data[sentinel][month] = image
    
    if nodata > 0:
      warnings.warn(f'[NODATA {nodata}] {s3path}')

  if nworkers < 2:
    for s3path in s3paths:
      assign(challenge.load_image(s3path))
  else:
    with ThreadPoolExecutor(nworkers) as executor:
      futures = [executor.submit(challenge.load_image, s3path) for s3path in s3paths]
      for future in as_completed(futures):
        assign(future.result())

  n = challenge.expire()
  # if n > 0:
  #   print(f'{n} images are expired')

  return chip_data

def preprocess_chip_data(challenge, chip_id):

  process_dir = challenge.subdir('processed')
  processed = process_dir/f'{chip_id}_feat.tif'

  s3paths_input, s3path_output = challenge.get_item(chip_id)

  if not processed.exists():

    chip_data = load_chip_data(challenge, s3paths_input)

    if False:
      import torch
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

      # concatenate S1 and S2 bands
      s1 = np.array(list(chip_data['s1'].values()))
      s2 = np.array(list(chip_data['s2'].values()))
      chip = np.concatenate((s1, s2), axis=0)

      # calculate mean for each band
      chip = torch.tensor(chip).to(device)
      chip = torch.mean(chip, dim=0).cpu().numpy()

    else:   
      # calculate average S1 for each band
      s1 = np.array(list(chip_data['s1'].values()))
      s1 = np.mean(s1, axis=0)

      # calculate average S2 for each band
      s2 = np.array(list(chip_data['s2'].values()))
      s2 = np.mean(s2, axis=0)

      # concatenate S1 and S2 bands
      chip = np.concatenate((s1, s2), axis=0)

    profile = dict(
      dtype='float32',
      compress='lzw',
      driver='GTiff'
    )

    save_raster(processed, chip, **profile)

    import time
    while not processed.exists():
      time.sleep(0.1)

  return str(processed), s3path_output

# #
# # segmentation transforms
# #
# import warnings
# import albumentations as A

# warnings.simplefilter('ignore')

# post_transform = A.Lambda(
#   name='post_transform',
#   image=post_transform,
#   mask=post_transform
# )

# train_transform = A.Compose([
#   A.RandomCrop(64, 64, p=1.),
#   A.Flip(p=0.75),
#   A.RandomBrightnessContrast(p=0.5),
#   post_transform
# ])

# valid_transform = A.Compose([
#     post_transform
# ])

# test_transform = A.Compose([
#     post_transform
# ])

# transforms = dict(
#   train_transform=train_transform,
#   valid_transform=valid_transform,
#   test_transform=test_transform
# )
