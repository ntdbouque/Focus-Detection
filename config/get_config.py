'''
Author: Nguyen Truong Duy
Purpose: Get detectron2 config
Latest Update: 19-02-2025
'''

from icecream import ic

from detectron2.config import get_cfg
from detectron2 import model_zoo
import yaml

def load_yaml_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def get_detectron2_config(yaml_path):
    '''
    Get detectron2 config
    
    Return:
    - cfg: detectron2 config
    '''
    
    yaml_config = load_yaml_config(yaml_path)
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(yaml_config['MODEL']['WEIGHTS']))
    cfg.DATASETS.TRAIN = (yaml_config['DATASETS']['TRAIN'],)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = yaml_config['DATALOADER']['NUM_WORKERS']
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(yaml_config['MODEL']['WEIGHTS']) 
    cfg.SOLVER.IMS_PER_BATCH = yaml_config['SOLVER']['IMS_PER_BATCH']
    cfg.SOLVER.BASE_LR = yaml_config['SOLVER']['BASE_LR']  
    cfg.SOLVER.MAX_ITER = yaml_config['SOLVER']['MAX_ITER']   
    cfg.SOLVER.STEPS = yaml_config['SOLVER']['STEPS']
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = yaml_config['MODEL']['ROI_HEADS']['BATCH_SIZE_PER_IMAGE']
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = yaml_config['MODEL']['ROI_HEADS']['NUM_CLASSES']
    cfg.MODEL.MASK_ON = yaml_config['MODEL']['MASK_ON']
    cfg.MODEL.DEVICE = yaml_config['MODEL']['DEVICE']
    # Inference at: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=7unkuuiqLdqd
    
    return cfg

def get_directory_config(yaml_path):
    '''
    Get directory config
    
    Return:
    - img_dir: image directory
    - label_dir: label directory
    - output_dir: output directory
    '''
    yaml_config = load_yaml_config(yaml_path)
    
    directory_cfg = {}
    
    directory_cfg['DATA_TRAIN_DIR'] = yaml_config['DIRECTORY']['DATA_TRAIN_DIR']
    directory_cfg['DATA_TEST_DIR'] = yaml_config['DIRECTORY']['DATA_TEST_DIR']
    directory_cfg['OUTPUT_DIR'] = yaml_config['DIRECTORY']['OUTPUT_DIR']
    directory_cfg['WEIGHT_DIR'] = yaml_config['DIRECTORY']['WEIGHT_DIR']
    directory_cfg['PREDICT_DIR'] = yaml_config['DIRECTORY']['PREDICT_DIR']
    return directory_cfg

detectron2_cfg = get_detectron2_config('config/config.yaml')
directory_cfg = get_directory_config('config/config.yaml')


if __name__ == '__main__':
    yaml_config = load_yaml_config('config/config.yaml')
    ic(yaml_config)