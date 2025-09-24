from lib.models.monodetr import build_monodetr, build_stereodetr

def build_model(cfg):

    if cfg["model_type"] == 'monodetr':
        return build_monodetr(cfg)
    elif cfg["model_type"] == 'stereodetr':
        return build_stereodetr(cfg)
    else:
        raise NotImplementedError("Model type '{}' not recognized.".format(cfg["model_type"]))
