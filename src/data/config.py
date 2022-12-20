import os
import yaml
import addict

from pathlib import Path

encoder = addict.Dict(
  names = [
    'efficientnet-b1',
    'efficientnet-b4',
    'inceptionresnetv2',
    'se_resnext50_32x4d',
    'timm-resnest14d'
  ],
  abbrs = [
    'effb1',
    'effb4',
    'inrv2',
    'srx50',
    'rns14'
  ]
)

enc2abb = {name: abbr for name, abbr in zip(*encoder.values())}

def _update_dict(d, p):
  for k, v in p.items():
    *path, key = k.split('.')
    inner_d = d
    for path_k in path:
      if inner_d[path_k] is None:
        inner_d[path_k] = {}
      inner_d = inner_d[path_k]
      old_v = inner_d.get(key)
      inner_d[key] = v

  return d

def _save(config, directory, name='config.yml'):
  os.makedirs(directory, exist_ok=True)
  fp = os.path.join(directory, name)
  with open(fp, 'w') as f:
    yaml.dump(config, f)

def _parse(**kwargs):
  cfg_path = kwargs['config']
  with open(cfg_path) as cfg:
    cfg_yaml = yaml.load(cfg, Loader=yaml.FullLoader)
  return _update_dict(cfg_yaml, kwargs)

def _train_cfg_template(stage, enc, fold, frac=1.):
  frac = '' if frac >= 1. else f'frac: {frac}'
  return f'''
logdir: models/{stage}/{enc2abb[enc]}-f{fold}/
gpus: [0]

# define model
model:
  architecture: Unet
  init_params:
    encoder_name: {enc}
    encoder_weights: imagenet
    classes: 1
    activation: identity
    in_channels: 15

data:
  df_path: ./data/image_folds.csv

  fold: {fold}

  # datasets
  train_dataset:
    name: SegmentationDataset
    init_params:
      transform_name: train_transform
      {frac}
    
  valid_dataset:
    name: SegmentationDataset
    init_params:
      transform_name: valid_transform
      {frac}

  # loaders
  train_dataloader:
    batch_size: 8 #6
    shuffle: true
    drop_last: true
    pin_memory: true
    num_workers: 4 #8

  valid_dataloader:
    batch_size: 16 #8
    shuffle: false
    drop_last: false
    pin_memory: true
    num_workers: 4 #16

training:

  losses:
    agbm:
      name: QuadraticLoss
      init_params:
    
  metrics:
    agbm:
    - name: AverageRMSE
      init_params:

  optimizer:
    name: Adam
    init_parmas:
      lr: 0.01 # 0.025 0.05 0.01 # 0.0001

  scheduler:
    name: PolyLR
    init_params:
      epochs: 50
  
  fit:
    epochs: 50
    accumulation_steps: 4
    verbose: true

  callbacks: []
'''

def _predict_cfg_template(stage, training_configs, fold=-1, frac=1.):
  frac = '' if frac >= 1. else f'frac: {frac}'

  return f'''
logdir: models/{stage}/predict/
gpus: [0]

#define model
training_configs: {training_configs}

data:
  df_path: ./data/image_folds.csv
  fold: {fold}

  dst_dir: ./data/prediction

  #datasets
  test_dataset:
    name: TestSegmentationDataset
    init_params:
      transform_name: test_transform
      {frac}

  #loaders
  test_dataloader:
    batch_size: 8
    shuffle: false
    drop_last: false
    pin_memory: true
    num_workers: 16
'''

class ConfigManager:
  def __init__(self, directory):
    self.base_dir = Path(directory).resolve()
    self.base_dir.mkdir(exist_ok=True)

  def _fill_train_config(self, challenge, stage, encs, frac=1.):
    for enc in encs:
      for fold in range(challenge.nfolds):
        # path = self.base_dir/f'{stage}-{enc2abb[enc]}-f{fold}.yml'
        # with path.open('w') as f:
        #   f.write(_train_cfg_template(stage, enc, self.stats, fold, frac))
        text = _train_cfg_template(stage, enc, fold, frac)
        config = yaml.load(text, Loader=yaml.FullLoader)
        config['challenge'] = challenge.to_dict()
        _save(config, self.base_dir, f'{stage}-{enc2abb[enc]}-f{fold}.yml')

  def _fill_predict_config(self, challenge, stage):
    train_cfgs = self.base_dir.glob(f'{stage}-*-f*.yml')
    train_cfgs = list(str(p) for p in sorted(
      list(p for p in train_cfgs),
      key=lambda p: p.stat().st_mtime_ns
    ))

    # path = self.base_dir/f'{stage}-predict.yml'
    # with path.open('w') as f:
    #   f.write(_predict_cfg_template(stage, train_cfgs, self.stats))
    text = _predict_cfg_template(stage, train_cfgs, frac=1.)
    config = yaml.load(text, Loader=yaml.FullLoader)
    config['challenge'] = challenge.to_dict()
    _save(config, self.base_dir, f'{stage}-predict.yml')

  def list_configs(self, pattern=None):
    return sorted(
      str(c) for c in self.base_dir.glob(pattern or '*')
    )

  def generate(self, challenge):
    self._fill_train_config(
      challenge,
      'stage1',
      [
        'efficientnet-b1',
        'se_resnext50_32x4d',
        'timm-resnest14d'
      ],
    )

    self._fill_predict_config(
      challenge,
      'stage1'
    )

  def next(self, stages=None):
    stages = stages or 'stage1'
    stages = stages if isinstance(stages, list) else [stages]
    for stage in sorted(stages):
      for path in self.list_configs(f'{stage}-*.yml'):
        cfg = addict.Dict(_parse(config=path))
        logdir = Path(cfg.logdir)
        if not (logdir/'__done__').exists():
          return path
    return None
