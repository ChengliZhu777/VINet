import models


def build_model(cfg):
    model_params = dict()

    for key in cfg:
        if key == 'type':
            continue
        model_params[key] = cfg[key]

    if cfg['type'] in ['VINet', 'OIDecoder']:
        return models.__dict__[cfg['type']](**model_params)
    elif cfg['type'] in ['VEBackbone']:
        return models.backbone.__dict__[cfg['type']](**model_params)
    else:
        raise KeyError
