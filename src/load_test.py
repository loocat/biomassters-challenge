def test_tiff_profile(filename):
  import rasterio
  with rasterio.open(filename) as fobj:
    dummy = fobj.profile
    fobj.read()

def main(args):
  from pathlib import Path
  from tqdm import tqdm

  paths = sorted(Path(args.dir).glob('**/*.tif'))
  for path in tqdm(paths):
    try:
      test_tiff_profile(path)
    except:
      print(path)
      path.unlink()

if __name__ == '__main__':
  import sys
  import addict
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--dir', type=str, required=True)
  parser.add_argument('--gpu', type=str, default='0')

  args = parser.parse_args()
  # print(args)
  # args = addict.Dict(args)
  main(args)

  sys.exit(0)