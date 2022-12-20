import os
import time
import boto3
import rasterio
import threading
import pandas as pd

from pathlib import Path
from pycksum import cksum
from botocore import UNSIGNED
from collections import deque, Counter
from botocore.config import Config

class DownloadCallback(object):

    def __init__(self, filename, size):
        self._filename = filename
        self._remain = size
        self._lock = threading.Lock()

    def __call__(self, bytes_amount):
        with self._lock:
            self._remain -= bytes_amount


class ChallengeCache:

    def __init__(self, directory, limit=100):
        self._s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
        self._directory = Path(directory)
        self._limit = limit
        self._files = deque()
        self._lock = threading.Lock()
        self.refresh()

    def refresh(self):
        with self._lock:
            for path in self._directory.glob('**/*'):
                if path.is_file():
                    self._files.append(str(path))

    def append(self, path):
        with self._lock:
            self._files.append(path)

    def exists(self, path):
        with self._lock:
            return path in self._files

    def get_file(self, s3path, chksum=None, size=0, refresh=False, ntries=3):
        tmp = s3path.split('/')
        if tmp[0] != 's3:':
            return s3path

        object_path = '/'.join(tmp[3:])
        local_path = self._directory / object_path

        if local_path.exists():
            if not refresh:
                return str(local_path)
            local_path.unlink()
            time.sleep(0.1)

        local_path.parent.mkdir(exist_ok=True)

        for _ in range(ntries):
            cb = DownloadCallback(str(local_path), size) if size > 0 else None
            with local_path.open('wb') as (fobj):
                bucket_name = tmp[2]
                self._s3.download_fileobj(bucket_name,
                  object_path,
                  fobj,
                  Callback=cb)

            while not local_path.exists():
                time.sleep(0.1)

            if chksum is None or chksum == cksum(open(local_path, 'rb')):
                self.append(str(local_path))
                return str(local_path)

        raise ValueError(f"Checksum Error: {s3path}")

    def expire(self):
        with self._lock:
            n = len(self._files)
            while len(self._files) > self._limit:
                f = self._files.popleft()
                if os.path.exists(f):
                    os.remove(f)

            return n - len(self._files)

    def cleanup(self):
        with self._lock:
            while len(self._files) > 0:
                f = self._files.popleft()
                if os.path.exists(f):
                    os.remove(f)


class ChallengeData(ChallengeCache):

    def __init__(self, config):
        super().__init__(config.base_dir + '/raw', 800)
        self._base = Path(config.base_dir)
        self._s3path = f"s3path_{config.s3region}"
        self._s3prefix = f"s3://{config.s3bucket}-{config.s3region}"
        self._id_str = config.id_str
        self._nodata = config.nodata

        self.df_meta = self._read_meta(self._s3prefix, config.metadata)

        df_fold_path = self._base / 'image_folds.csv'
        if df_fold_path.exists():
            self.df_fold = pd.read_csv(df_fold_path, index_col=0)
        else:
            self.df_fold = self._make_folds(config.nfolds)
            self.df_fold.to_csv(df_fold_path)

    def resolve(self, path):
        path = path if isinstance(path, Path) else Path(path)
        if len(path.parts) > 1:
            return path.resolve()
        else:
            return (self._base / path).resolve()

    def _read_meta(self, s3prefix, metadata):
        df = {}
        for k in ('features', 'train_agbm'):
            meta = self.get_file(f"{s3prefix}/{metadata[k]}")
            df[k] = pd.read_csv(meta)

        if metadata.features_stats:
            latest_stats = self.resolve(metadata.features_stats)
            if latest_stats.exists():
                stats = pd.read_csv(latest_stats).drop((self._s3path), axis=1)
                stats = stats.loc[(stats.defects > -1)]
                feats = df['features'].loc[stats.index]
                df['features'] = feats.join((stats.set_index('filename')),
                  on='filename')
        return df

    def _make_folds(self, nfolds):
        df_feat = self.df_meta['features']
        df_agbm = self.df_meta['train_agbm']

        if 'defects' in df_feat.columns:
            df_images = df_feat.loc[((df_feat.defects == 0) | (df_feat.split == 'test'))]
        else:
            df_images = df_feat

        chip_ids = df_images.groupby(['chip_id', 'satellite']).count() > 0
        chip_ids = Counter(id for id, _ in chip_ids.index)
        chip_ids = [id for id, cnt in chip_ids.items() if cnt == 2]

        df_feat = df_feat.loc[df_feat.chip_id.isin(chip_ids)]

        train_ids = df_feat.loc[(df_feat.split == 'train')][self._id_str].unique()
        train_ids = list(set.intersection(set(train_ids), set(df_agbm[self._id_str].unique())))
        print(len(train_ids), 'training chips')

        test_ids = df_feat.loc[(df_feat.split == 'test')][self._id_str].unique()
        print(len(test_ids), 'testing chips')

        df = pd.DataFrame({'fold': -1}, index=train_ids)
        df = df.sample(frac=1.0, random_state=42)

        size = (df.shape[0] + nfolds - 1) // nfolds
        for fold in range(nfolds):
            df.iloc[fold * size:(fold + 1) * size, 0] = fold

        return pd.concat([df, pd.DataFrame({'fold': -1}, index=test_ids)])

    def get_means(self, chip_ids=None):
        if chip_ids is None:
            chip_ids = self.df_fold.index.values

        df = self.df_meta['features']
        df = df.loc[df.chip_id.isin(chip_ids)]

        means = {}
        for sentinel in ('S1', 'S2'):
            cols = [f"mean_b{i}" for i in range(len(self._nodata[sentinel.lower()]))]
            vals = df.loc[(df.satellite == sentinel, cols)].mean().values
            means[sentinel.lower()] = vals

        return means

    def load_image(self, s3path, chksum=None, size=0, refresh=False):
        tmp = s3path.split('/')
        if tmp[0] == 's3':
            filename = tmp[(-1)]
            if filename.lower().endswith('_agbm.tif'):
                df = self.df_meta['train_agbm']
            else:
                df = self.df_meta['features']

            if size < 1:
                values = df.loc[(df.filename == filename, 'size')].values
                if len(values) == 1:
                    size = values[0]
                else:
                    raise ValueError(f"Unknown file size: {filename}")
                    
            if chksum is None:
                values = df.loc[(df.filename == filename, 'cksum')].values
                if len(values) == 1:
                    chksum = values[0]

        raw_path = None
        while raw_path is None or not os.path.exists(raw_path):
            raw_path = self.get_file(s3path, chksum, size, refresh)

        with rasterio.open(raw_path) as (file):
            return (file.read(), raw_path)

    def get_stats(self, names=None, chip_ids=None):
        names = ['mean', 'std'] if names is None else names
        names = [names] if isinstance(names, str) else names

        if chip_ids is None:
            chip_ids = self.df_fold.index.values

        df = self.df_meta['features']
        df = df.loc[df.chip_id.isin(chip_ids)]

        stats = {}
        for name in names:
            stat = {}
            for sentinel in ('s1', 's2'):
                cols = [f"{name}_b{i}" for i in range(len(self._nodata[sentinel]))]
                rows = df.satellite == sentinel.upper()
                if df.loc[(rows, cols)].size > 0:
                    vals = df.loc[(rows, cols)].mean().values
                    stat[sentinel] = list(float(v) for v in vals)

            stats[name] = stat

        return stats

    def get_ids(self, fold):
        return self.df_fold.loc[(self.df_fold.fold == fold)].index

    def get_item(self, id):
        df_feat = self.df_meta['features']
        df_agbm = self.df_meta['train_agbm']

        s3paths_feat = df_feat.loc[(df_feat[self._id_str] == id)][self._s3path].values
        if self.df_fold.at[(id, 'fold')] != -1:
            s3paths_agbm = df_agbm.loc[(df_agbm[self._id_str] == id)][self._s3path].values
        else:
            s3paths_agbm = None

        return (s3paths_feat, s3paths_agbm[0] if s3paths_agbm else None)