import models


def build_model(cfg, build_type='model'):
    model_params = dict()

    for key in cfg:
        if key == 'type':
            continue
        model_params[key] = cfg[key]

    return models.__dict__[cfg['type']](**model_params)
  
