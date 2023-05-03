import sklearn # prevent error: cannot allocate memory in static TLS block

import os
import fire
import ttach
import torch
import addict
import argparse
import rasterio
import pandas as pd

from tqdm import tqdm
from multiprocessing import Pool

from .training.config import parse_config
from .training.runner import NormRunner

from . import getters


class EnsembleModel(torch.nn.Module):
    """Ensemble of torch models, pass tensor through all models and average results"""

    def __init__(self, models: list):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        result = None
        for model in self.models:
            y = model(x)
            if result is None:
                result = y
            else:
                result += y
        result /= torch.tensor(len(self.models)).to(result.device)
        return result



def model_from_config(path: str, device):
    cfg = addict.Dict(parse_config(path))
    init_params = cfg.model.init_params
    init_params.encoder_weights = None
    model = getters.get_model(
        architecture=cfg.model.architecture,
        init_params=init_params
    )
    checkpoint_path = os.path.join(cfg.logdir, 'checkpoints', 'best.pth')
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location=device)
        print(checkpoint_path)
        model.load_state_dict(state_dict['state_dict'])
        return model
    return None

def get_agbm_profile(challenge):
    chip_id = challenge.get_ids(0)[0]
    _, agbm = challenge.get_item(chip_id)
    agbm = challenge.get_file(agbm)
    with rasterio.open(agbm) as f:
        return f.profile

def write_agbm(dst_path, agbm, profile, **kwargs):
    c, h, w = agbm.shape
    profile.update(dict(
        height=h,
        width=w,
        count=c
    ))
    profile.update(**kwargs)
    with rasterio.open(dst_path, 'w', **profile) as dst:
        dst.write(agbm)

def main(args):

    from .data.challenge import ChallengeData
    challenge = ChallengeData(args.challenge)

    # set GPUS
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.gpus)) if args.get("gpus") else ""
    

    # --------------------------------------------------
    # define model
    # --------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Available devices:", device)

    # loading trained models
    models = [model_from_config(config_path, device) for config_path in args.training_configs]
    models = list(model for model in models if model is not None)

    # create ensemble
    model = EnsembleModel(models)

    # add test time augmentations (flipping and rotating input image)
    model = ttach.SegmentationTTAWrapper(
        model, ttach.aliases.d4_transform(),
        merge_mode='mean'
    )

    # create Multi GPU model if number of GPUs is more than one
    if len(args.gpus) > 1:
        model = torch.nn.DataParallel(model, args.gpus)

    print("Done loading...")
    model.to(device)

    # --------------------------------------------------
    # start evaluation
    # --------------------------------------------------
    runner = NormRunner(
        model,
        model_device=device,
        stats=challenge.get_stats(names=['mean', 'std'])
    )
    model.eval()


    # test dataloader
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        **args.data.test_dataloader
    )

    agbm_profile = get_agbm_profile(challenge)

    for batch in tqdm(test_dataloader):
        ids = batch['id']
        preds = runner.predict_on_batch(batch)['agbm']

        for chip_id, pred in zip(ids, preds):
            pred = pred.cpu().numpy().astype('float32')
            dst_path = os.path.join(args.data.dst_dir, f'{chip_id}_agbm.tif')
            write_agbm(dst_path, pred, agbm_profile, compress='lzw', driver='GTiff')

if __name__ == "__main__":

    cfg = addict.Dict(fire.Fire(parse_config))
    main(cfg)