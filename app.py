'''
Author: Nguyen Truong Duy
Purpose: API inference with detectron2 model
Latest Update: 02-03-2025
'''

import gradio as gr
import torch
from source.predict import detectron_infer

from source.utils import visualize

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

import os
os.environ["GRADIO_TEMP_DIR"] = "/workspace/competitions/Sly/detectron2_train_infer/temp_gradio"

output_path = '/workspace/competitions/Sly/detectron2_train_infer/temp_gradio/output.mp4'
checkpoint_path = '/workspace/competitions/Sly/detectron2_train_infer/output/model_0004999.pth'

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detectron_infer(img):
    '''
    Inferencing with detectron2 model
    Args:
        checkpoint_path (str): Path to model checkpoint
        img_path (str): Path to image file
    Returns:    
        dct_result (dict): Dictionary of inference result
    '''
    dct_result = {}
    
    #cfg = get_cfg()

    # #cfg.merge_from_file(model_zoo.get_config_file(detectron2_cfg['MODEL']['WEIGHTS']))
    # cfg.MODEL.ROI_HEADS.NUM_CLASSES = detectron2_cfg['MODEL']['ROI_HEADS']['NUM_CLASSES']
    # cfg.MODEL.WEIGHTS = checkpoint_path 
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0
    # cfg.MODEL.DEVICE = "cuda"  
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

    cfg.MODEL.WEIGHTS = '/workspace/competitions/Sly/detectron2_train_infer/output/model_0004999.pth'  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    
    predictor = DefaultPredictor(cfg)
    
    outputs = predictor(img)
    pred_classes = outputs["instances"].pred_classes.cpu().numpy()
    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    scores = outputs["instances"].scores.cpu().numpy()
    
    visualized_image = visualize(img, pred_boxes, scores, pred_classes)
    
    return visualized_image

def detectron_infer_video(video_path):
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("my_dataset_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
    cfg.SOLVER.STEPS = []        # do not decay learning rate
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 8   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)

    cfg.MODEL.WEIGHTS = '/workspace/competitions/Sly/detectron2_train_infer/output/model_0004999.pth'  # path to the model we just trained
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    
    predictor = DefaultPredictor(cfg)
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        outputs = predictor(frame)
        
        pred_classes = outputs["instances"].pred_classes.cpu().numpy()
        pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()
        
        frame = visualize(frame, pred_boxes, scores, pred_classes, is_save=False)
        out.write(frame)
    
    cap.release()
    out.release()
    return output_path

with gr.Blocks() as demo:
    gr.Markdown("## Detectron2 Inference")

    with gr.Tabs():
        with gr.TabItem("Image Inference"):
            img_input = gr.Image(type="numpy")
            img_output = gr.Image()
            img_btn = gr.Button("Infer Image")
            img_btn.click(fn=detectron_infer, inputs=img_input, outputs=img_output)

        with gr.TabItem("Video Inference"):
            vid_input = gr.Video(label="Upload Video")
            vid_output = gr.Video()
            vid_btn = gr.Button("Infer Video")
            vid_btn.click(fn=detectron_infer_video, inputs=vid_input, outputs=vid_output)

# Chạy Gradio
if __name__ == "__main__":
    demo.launch(share=True)