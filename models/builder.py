import models


def build_model(cfg):
    model_params = dict()

    for key in cfg:
        if key == 'type':
            continue
        model_params[key] = cfg[key]

    if cfg['type'] in ['VINet']:
        return models.__dict__[cfg['type']](**model_params)
    elif cfg['type'] in ['VEBackbone']:
        return models.backbone.__dict__[cfg['type']](**model_params)
    elif cfg['type'] in ['RVIModule']:
        return models.module.__dict__[cfg['type']](**model_params)
    elif cfg['type'] in ['TransformerRecognitionHead']:
        return models.  recognizer.__dict__[cfg['type']](**model_params)
    else:
        raise KeyError
