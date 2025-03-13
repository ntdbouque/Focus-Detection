'''
Author: Nguyen Truong Duy
Purpose: Evaluate the performance of the Detecon2 model
Latest Update: 19-02-2025
'''
import os
import sys
from pathlib import Path
from icecream import ic
sys.path.append(str(Path(__file__).resolve().parent.parent))

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2.engine import DefaultPredictor

from config.get_config import detectron2_cfg
from source.utils import register_detectron2_dataset

class Evaluator():
    def __init__(self, checkpoint_path, output_dir, selective_class=None, cfg=detectron2_cfg):
        self.cfg = cfg
        # inference at: https://colab.research.google.com/drive/16jcaJoc6bCFAQ96jDe2HwtXj7BMD_-m5#scrollTo=Ya5nEuMELeq8
        self.cfg.MODEL.WEIGHTS = os.path.join(checkpoint_path) 
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
        
        self.output_dir = output_dir
        self.selective_class = selective_class
        self.predictor = DefaultPredictor(self.cfg)
        
    def evaluate(self):
        '''
        Evaluate the performance of the Detecon2 model
        
        Returns:
            dict: A dictionary containing the evaluation results
        '''
        
        register_detectron2_dataset(type='test', img_dir = './data/valid', selective_classes=self.selective_class)
        
        evaluator = COCOEvaluator("my_dataset_test", output_dir=self.output_dir)
        val_loader = build_detection_test_loader(self.cfg, "my_dataset_test")
        results = inference_on_dataset(self.predictor.model, val_loader, evaluator)
        ic(results)
        
    def save_prediction_on_test(self):
        '''
        Save drawed images with bboxes, class names, scores on image
        '''
        pass
        
if __name__ == '__main__':
    evaluator = Evaluator('/workspace/competitions/Sly/detectron2_train_infer/output/model_0009999.pth', './output', selective_class=[0,1])
    evaluator.evaluate()