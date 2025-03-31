'''
Author: Nguyen Truong Duy
Purpose: Training detectron2 model
Latest Update: 17-02-2025
'''


import os, sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import os, json, cv2, random
import torch

from icecream import ic

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor, DefaultTrainer, HookBase
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

from source.utils import register_detectron2_dataset, clear_directory
from config.get_config import detectron2_cfg, directory_cfg


class PrintEpochHook(HookBase):
    def __init__(self, total_epochs, iterations_per_epoch):
        self.total_epochs = total_epochs
        self.iterations_per_epoch = iterations_per_epoch

    def after_step(self):
        current_iteration = self.trainer.iter + 1  
        current_epoch = current_iteration // self.iterations_per_epoch + 1
        if current_iteration % self.iterations_per_epoch == 0 or current_iteration == self.trainer.max_iter:
            print(f"Đã hoàn thành epoch: {current_epoch}/{self.total_epochs} - Iteration: {current_iteration}/{self.trainer.max_iter}")


def detectron_train():
    '''
    Purpose: training a detectron model
    Flow: RegisterDataset -> Detectron2 Configuration -> Training -> save checkpoint
    Args:
        device: 'cuda' or 'cpu'
        selective_classes: list of classes to train
        num_class: number of classes in dataset
    '''

    clear_directory()
    ic(directory_cfg)
        
    num_images = 0
    for file in os.listdir(os.path.join(directory_cfg['DATA_TRAIN_DIR'], 'images')):
        if file.lower().endswith(('.jpg', '.png')):
            num_images += 1
    ic(num_images)
    
    register_detectron2_dataset(img_dir=directory_cfg['DATA_TRAIN_DIR'], type='train')
    
    iterations_per_epoch = num_images // detectron2_cfg['SOLVER']['IMS_PER_BATCH']
    total_epochs = detectron2_cfg['SOLVER']['MAX_ITER'] // iterations_per_epoch
    
    os.makedirs(directory_cfg['OUTPUT_DIR'], exist_ok=True)

    trainer = DefaultTrainer(detectron2_cfg)
    trainer.resume_or_load(resume=False)
    trainer.register_hooks([PrintEpochHook(total_epochs, iterations_per_epoch)])  
    trainer.train()
    
    
    final_model_path = os.path.join(directory_cfg['WEIGHT_DIR'], 'best.pth')
    torch.save(trainer.model.state_dict(), final_model_path)


if __name__ == '__main__':    
    
    device = 'cuda'
    
    detectron_train(device)
    

