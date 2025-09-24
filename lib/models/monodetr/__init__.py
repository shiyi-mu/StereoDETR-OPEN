from .monodetr import build
from .stereodetr import build_StereoDETR

def build_monodetr(cfg):
    return build(cfg)

def build_stereodetr(cfg):
    return build_StereoDETR(cfg)





