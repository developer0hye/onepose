import numpy as np
import cv2
import torch
import torch.nn as nn
import onepose.models.vitpose as vitpose
from PIL import Image

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
    def forward(self, x: Union[np.ndarray, Image.Image, List[Union[np.ndarray, Image.Image]]]) -> Union[Dict, List[Dict]]:
        if self.training:
            self.eval()
        
        device = next(self.parameters()).device
        
        single_image = False
        # Input validation and conversion
        if isinstance(x, list):
            if not x:  # empty list check
                raise ValueError("Input list cannot be empty")
            if not all(isinstance(img, (np.ndarray, Image.Image)) for img in x):
                raise TypeError("All elements in the list must be either numpy arrays or PIL Images")
            # Convert PIL images to numpy arrays with BGR color space
            x = [cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR) if isinstance(img, Image.Image) else img for img in x]
        elif isinstance(x, (np.ndarray, Image.Image)):
            if isinstance(x, Image.Image):
                x = cv2.cvtColor(np.array(x), cv2.COLOR_RGB2BGR)
            x = [x]
            single_image = True
        else:
            raise TypeError("Input must be either a numpy array, PIL Image, or a list of them")
        
        # Convert grayscale images to BGR
        for i, img in enumerate(x):
            if img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 1):
                x[i] = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        batch_results = []
        for img in x:
            img_height, img_width = img.shape[:2]
            center, scale = _box2cs(self.model_cfg.data_cfg['image_size'], 
                                [0, 0, img_width, img_height])      

            results = {
                'img': img,
                'rotation': 0,
                'center': center,
                'scale': scale,
                'image_size': np.array(self.model_cfg.data_cfg['image_size']),
            }

            results = self.transforms(results)
            batch_results.append(results['img'])

        # Stack transformed images into a batch
        batch_tensor = torch.stack(batch_results).to(device)
        
        # Forward pass
        out = self.model(batch_tensor)
        out = out.cpu().numpy()
        
        # Process each image's predictions
        centers = [_box2cs(self.model_cfg.data_cfg['image_size'], 
                        [0, 0, img.shape[1], img.shape[0]])[0] for img in x]
        scales = [_box2cs(self.model_cfg.data_cfg['image_size'], 
                        [0, 0, img.shape[1], img.shape[0]])[1] for img in x]
        
        points, maxvals = keypoints_from_heatmaps(
            out,
            center=centers,
            scale=scales,
            unbiased=False,
            post_process='default',
            kernel=11,
            valid_radius_factor=0.0546875,
            use_udp=self.use_udp,
            target_type='GaussianHeatmap'
        )
        
        outputs = [{'points': p, 'confidence': c} for p, c in zip(points, maxvals)]
        
        # Return single result for single image input
        if single_image:
            return outputs[0]
        return outputs

def create_model(model_name: str = 'ViTPose_huge_simple_coco') -> Model:
    model = Model(model_name=model_name)
    return model

def list_models() -> List[str]:
    return list(model_config.keys())
