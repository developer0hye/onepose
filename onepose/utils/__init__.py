import hashlib
import urllib.request
from tqdm import tqdm
import os
import pathlib
import importlib.util

def calc_file_hash(path):
    f = open(path, 'rb')
    data = f.read()
    hash = hashlib.md5(data).hexdigest()
    return hash

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_weights(url: str, filename: str, hash: str) -> None:
    if os.path.exists(filename):
        if calc_file_hash(filename) == hash or hash == 'test':
            print(f'{filename} is already downloaded.')
            return True
        os.remove(filename)

    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)

    if hash is not None:
        assert hash == calc_file_hash(filename), f'{filename} is corrupted. Please run code again.'

def read_cfg(cfg):
    spec = importlib.util.spec_from_file_location("config", cfg)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    return config_module