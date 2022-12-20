import sys
import torch
import torchvision

import addict
import argparse

def test_tiff_profile(filename):
  import rasterio
  with rasterio.open(filename) as fobj:
    print(fobj.profile)

def main(args):
  test_tiff_profile(
    challenge.get_file('s3://drivendata-competition-biomassters-public-as/train_agbm/00301627_agbm.tif')
  )

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # parser.add_argument(
  #   '--configs', 
  # )

  args = parser.parse_args()
  main(args)
  sys.exit(0)