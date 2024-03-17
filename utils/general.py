import os
import re
import yaml
import glob
import logging
import argparse

from pathlib import Path


def set_logging(rank=-1):
    """
    %(message)s : output log content
    %(asctime)s : output logging time
    """
    logging.basicConfig(format='%(message)s',
                        level=logging.INFO if rank in [-1, 0] else logging.WARN)


def get_options(hyp_path='hyp/VINet.base.yaml',
                data_path='data/VINet.yaml',
                model_cfg='models/cfg/VINet.yaml',
                resume=False, project='runs', name='Train-1'):

    parser = argparse.ArgumentParser(description='VINet Train/Test Options')
    parser.add_argument('--hyp', type=str, default=hyp_path, help='Required hyper-parameters, i.e, batch-size.')
    parser.add_argument('--data', type=str, default=data_path, help='Train/Test dataset paths.')
    parser.add_argument('--model-cfg', type=str, default=model_cfg, help='VINet architecture.')
    parser.add_argument('--weights', type=str, default='', help='trained weight path.')
    parser.add_argument('--resume', default=resume, help='resume most recent training.')
    parser.add_argument('--project', default=project, help='save to project/name')
    parser.add_argument('--name', default=name, help='save to project/name')

    return parser.parse_args()


def get_latest_run(search_dir='.'):
    last_ckpt_list = glob.glob(f'{search_dir}/**/last*.pt', recursive=True)
    if last_ckpt_list:
        return max(last_ckpt_list, key=os.path.getctime)
    else:
        raise FileNotFoundError("Error: don't find checkpoint file for resuming train.")


def check_filepath(path):
    if path is None or path == '':
        raise ValueError('Error: empty filename, specify exact file path.')
    elif Path(path).is_file():
        return path
    else:
        files = glob.glob('./**/' + path, recursive=True)
        assert len(files) >= 1, f"Error: file '{path}' not found."
        assert len(files) == 1, f"Error: Multiple files match '{path}', specify exact filepath in: {files}."
        return files[0]


def increment_path(path, exist_ok=True, separator='-'):
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return path.absolute()
    else:
        paths = glob.glob(f'{path.parent}/{path.stem.split(separator)[0]}{separator}*')
        match_pattern = rf"%s{separator}(\d+)" % path.stem.split(separator)[0]

        matches = [re.search(match_pattern, p) for p in paths]
        i = [int(m.groups()[0]) for m in matches if m]

        return f"{path.parent}/{path.stem.split(separator)[0]}{separator}{max(i) + 1 if i else 2}"


def colorstr(*inputs):
    *args, string = inputs if len(inputs) > 1 else ('blue', 'bold', inputs[0])  # color arguments, string
    colors = {'black': '\033[30m',  # basic colors
              'red': '\033[31m',
              'green': '\033[32m',
              'yellow': '\033[33m',
              'blue': '\033[34m',
              'magenta': '\033[35m',
              'cyan': '\033[36m',
              'white': '\033[37m',
              'bright_black': '\033[90m',  # bright colors
              'bright_red': '\033[91m',
              'bright_green': '\033[92m',
              'bright_yellow': '\033[93m',
              'bright_blue': '\033[94m',
              'bright_magenta': '\033[95m',
              'bright_cyan': '\033[96m',
              'bright_white': '\033[97m',
              'end': '\033[0m',  # misc
              'bold': '\033[1m',
              'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


def load_file(file_path):
    try:
        with open(file_path) as f:
            return yaml.load(f, yaml.SafeLoader)
    except Exception as e:
        raise Exception(f'Error: fail to load data from {file_path}. \n {e}')
