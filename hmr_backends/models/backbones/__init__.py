from .vit import vit
from .vit_wilor import vit_wilor

def create_backbone(cfg):
    if cfg.MODEL.BACKBONE.TYPE == 'vit':
        return vit(cfg)
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_wilor':
        return vit_wilor(cfg)
    else:
        raise NotImplementedError('Backbone type is not implemented')
