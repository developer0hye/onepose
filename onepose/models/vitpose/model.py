import torch.nn as nn

from onepose.models.vitpose.backbone import ViT
from onepose.models.vitpose.head import TopdownHeatmapSimpleHead


__all__ = ['ViTPose']


class ViTPose(nn.Module):
    def __init__(self, cfg: dict) -> None:
        super(ViTPose, self).__init__()
        
        backbone_cfg = {k: v for k, v in cfg['backbone'].items() if k != 'type'}
        head_cfg = {k: v for k, v in cfg['keypoint_head'].items() if k != 'type'}
        
        self.backbone = ViT(**backbone_cfg)
        self.keypoint_head = TopdownHeatmapSimpleHead(**head_cfg)
    
    def forward_features(self, x):
        return self.backbone(x)
    
    def forward(self, x):
        features = self.forward_features(x)
        return self.keypoint_head(features)
