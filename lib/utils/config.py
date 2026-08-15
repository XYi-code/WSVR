import os
import yaml
import yaml
from yaml import SafeLoader

from easydict import EasyDict as edict

config = edict()

config.CONFIG_NAME = ''
config.OUTPUT_DIR = ''
config.DATA_DIR = ''
config.CUDA = True
config.WORKERS = 1
config.PRINT_FREQ = 20

config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

config.MLP = edict()
config.MLP.TYPE = 'SMLC'

config.MODEL = edict()
config.MODEL.NAME = 'resnet50'
config.MODEL.INIT_WEIGHTS = True
config.MODEL.PRETRAINED = ''
config.MODEL.IMAGE_SIZE = [256, 256]  
config.MODEL.FEATURES = 2048
config.MODEL.CLASSES = 751

config.WSVR = edict()

config.WSVR.K_CENTERS = 2  

config.WSVR.ALPHA_SCL = 0.5   
config.WSVR.TEMP = 0.1     

config.WSVR.LAMBDA_1 = 0.2   
config.WSVR.LAMBDA_2 = 1.0    

config.WSVR.T = 0.5
config.WSVR.ICE_HARD_K = 32

config.DATASET = edict()
config.DATASET.ROOT = ''
config.DATASET.DATASET = ''
config.DATASET.DATA_FORMAT = 'jpg'
config.DATASET.CAM_NUM = 2

config.DATASET.RE = 0.5
config.DATASET.NAMES = 'AK'
config.DATASET.ROOT_DIR = ''

config.TRAIN = edict()
config.TRAIN.LR = 0.1
config.TRAIN.LR_STEP = 40
config.TRAIN.LR_FACTOR = 0.1

config.TRAIN.OPTIMIZER = 'sgd'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WEIGHT_DECAY = 0.0005
config.TRAIN.NESTEROV = True

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 40

config.TRAIN.RESUME = True
config.TRAIN.CHECKPOINT = ''

config.TRAIN.BATCH_SIZE = 128
config.TRAIN.SHUFFLE = True

config.TEST = edict()
config.TEST.BATCH_SIZE = 32
config.TEST.MODEL_FILE = ''
config.TEST.OUTPUT_FEATURES = 'pool5'

def _update_dict(k, v):
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))

def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = yaml.load(f, Loader=SafeLoader)
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))
    
    if 'DATASET' in exp_config and 'NAMES' not in exp_config['DATASET']:
        config.DATASET.NAMES = config.DATASET.DATASET

def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)

if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])
