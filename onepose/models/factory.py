import numpy as np
import torch
import torch.nn as nn
import onepose.models.vitpose as vitpose
from onepose.utils import read_cfg, download_weights
from onepose.transforms import ComposeTransforms, BGR2RGB, TopDownAffine, ToTensor, NormalizeTensor, _box2cs
from onepose.functional import keypoints_from_heatmaps
import os
import pathlib
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Sequence, Union, Tuple

model_config = {
    'ViTPose_base_simple_coco': {
        'model_cfg': 'ViTPose_base_simple_coco_256x192.py',
        'dataset_cfg': 'coco.py',
        'url': 'https://github.com/developer0hye/onepose/releases/download/1.0.0/vitpose-b-simple_half.pth',
        'hash': '0408c829e344fe6f9d61eb16db5c863f'
    },
    'ViTPose_large_simple_coco': {
        'model_cfg': 'ViTPose_large_simple_coco_256x192.py',
        'dataset_cfg': 'coco.py',
        'url': 'https://github.com/developer0hye/onepose/releases/download/1.0.0/vitpose-l-simple_half.pth',
        'hash': '6b35d98cdf0ac4838dbe9f4bb98dd38f'
    },
    'ViTPose_huge_simple_coco': {
        'model_cfg': 'ViTPose_huge_simple_coco_256x192.py',
        'dataset_cfg': 'coco.py',
        'url': 'https://github.com/developer0hye/onepose/releases/download/1.0.0/vitpose-h-simple_half.pth',
        'hash': '319c1bf933f677bce2ad33da21304866'
    },
    'ViTPose_base_mpii': {
        'model_cfg': 'ViTPose_base_mpii_256x192.py',
        'dataset_cfg': 'mpii.py',
        'url': 'https://github.com/developer0hye/onepose/releases/download/1.0.0/vitpose-b-multi-mpii_half.pth',
        'hash': '475eaab9c8fd78df77729cac7229c3e7'
    },
    'ViTPose_large_mpii': {
        'model_cfg': 'ViTPose_large_mpii_256x192.py',
        'dataset_cfg': 'mpii.py',
        'url': 'https://github.com/developer0hye/onepose/releases/download/1.0.0/vitpose-l-multi-mpii_half.pth',
        'hash': '1c7a3a6d40e775b2ca376090bf8f55ed'
    },
    'ViTPose_huge_mpii': {
        'model_cfg': 'ViTPose_huge_mpii_256x192.py',
        'dataset_cfg': 'mpii.py',
        'url': 'https://github.com/developer0hye/onepose/releases/download/1.0.0/vitpose-h-multi-mpii_half.pth',
        'hash': '38a0335fbc749c1bfb6b60f0b13e5c93'
    },
    'ViTPose+_small_coco_wholebody': {
        'model_cfg': 'ViTPose_small_wholebody_256x192.py',
        'dataset_cfg': 'coco_wholebody.py',
        'url': 'https://github.com/developer0hye/onepose/releases/download/1.0.0/splitted_vitpose+_small_coco_wholebody_half.pth',
        'hash': 'cfcd5161321e4ed8c14c9fd62e3655af'
    },
    'ViTPose+_base_coco_wholebody': {
        'model_cfg': 'ViTPose_base_wholebody_256x192.py',
        'dataset_cfg': 'coco_wholebody.py',
        'url': 'https://github.com/developer0hye/onepose/releases/download/1.0.0/splitted_vitpose+_base_coco_wholebody_half.pth',
        'hash': 'f454d7d10dc15325f072d56bd658760e'
    },
    'ViTPose+_large_coco_wholebody': {
        'model_cfg': 'ViTPose_large_wholebody_256x192.py',
        'dataset_cfg': 'coco_wholebody.py',
        'url': 'https://github.com/developer0hye/onepose/releases/download/1.0.0/splitted_vitpose+_large_coco_wholebody_half.pth',
        'hash': 'cb66cedb6ac06bdd8ed9ca3b9c4cf8b7'
    },
    'ViTPose+_huge_coco_wholebody': {
        'model_cfg': 'ViTPose_huge_wholebody_256x192.py',
        'dataset_cfg': 'coco_wholebody.py',
        'url': 'https://github.com/developer0hye/onepose/releases/download/1.0.0/splitted_vitpose+_huge_coco_wholebody_half.pth',
        'hash': '8b9bea9e0377561c201fd9e0f0973afe'
    },
}

class Model(nn.Module):
    def __init__(self, 
                 model_name: str = 'ViTPose_huge_simple_coco') -> None:
        super().__init__()

        file_path = pathlib.Path(os.path.abspath(__file__)).parent
        
        self.model_cfg = read_cfg(os.path.join(file_path, 'configs', model_config[model_name]['model_cfg']))
        self.model = vitpose.ViTPose(self.model_cfg.model)

        self.use_udp = self.model_cfg.model['test_cfg'].get('use_udp', False)
        self.transforms = ComposeTransforms([
            BGR2RGB(),
            TopDownAffine(use_udp=self.use_udp),
            ToTensor(),
            NormalizeTensor()
        ])

        weights_folder = os.path.join(file_path, 'weights')
        os.makedirs(weights_folder, exist_ok=True)
        ckpt = os.path.join(weights_folder, model_config[model_name]['url'].split('/')[-1])
        download_weights(model_config[model_name]['url'], 
                         ckpt, 
                         model_config[model_name]['hash'])
        self.model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        self.model.eval()
        
        dataset_cfg = read_cfg(os.path.join(file_path.parent, 'datasets', model_config[model_name]['dataset_cfg']))
        self.keypoint_info = dataset_cfg.dataset_info['keypoint_info']
        self.skeleton_info = dataset_cfg.dataset_info['skeleton_info']

    @torch.no_grad()
    @torch.inference_mode()
    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.training:
            self.eval()
        
        device = next(self.parameters()).device
        
        img_height, img_width = x.shape[:2]
        center, scale = _box2cs(self.model_cfg.data_cfg['image_size'], [0, 0, img_width, img_height])      

        results = {'img': x,
                   'rotation': 0,
                   'center': center,
                   'scale': scale,
                   'image_size': np.array(self.model_cfg.data_cfg['image_size']),
                   }

        results = self.transforms(results)
        results['img'] = results['img'].to(device)
        
        out = self.model(results['img'][None, ...])
        out = out.cpu().numpy()
        
        out, maxvals = keypoints_from_heatmaps(out, 
                                      center=[center],
                                      scale=[scale], 
                                      unbiased=False, 
                                      post_process='default', 
                                      kernel=11, 
                                      valid_radius_factor=0.0546875, 
                                      use_udp=self.use_udp, 
                                      target_type='GaussianHeatmap')
        out = out[0]
        maxvals = maxvals[0]
        out = {'points': out, 'confidence': maxvals}
        return out

def create_model(model_name: str = 'ViTPose_huge_simple_coco') -> Model:
    model = Model(model_name=model_name)
    return model

def list_models() -> List[str]:
    return list(model_config.keys())
