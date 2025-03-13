'''
Author: Nguyen Truong Duy
Purpose: Utility functions
Latest Update: 19-02-2025
'''

import sys
import os
from pathlib import Path
from icecream import ic
sys.path.append(str(Path(__file__).resolve().parent.parent))

from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode

from config.get_config import detectron2_cfg, directory_cfg
import cv2

def clear_directory():
    '''
    Clear directory
    '''
    os.system(f'rm -rf {directory_cfg["OUTPUT_DIR"]}/*')
    ic(f'Cleared {directory_cfg["OUTPUT_DIR"]}')

    os.system(f'rm -rf {directory_cfg["WEIGHT_DIR"]}/*')
    ic(f'Cleared {directory_cfg["WEIGHT_DIR"]}')
    
def visualize(img, bboxes, scores, classes_name, is_save=True):
    '''
    Visualize and saved inference result
    Args:
        img (np.array): Image
        bboxes (list): List of bounding boxes
        scores (list): List of scores
        classes_name (list): List of class names
        is_save (bool): Save image or not
    Returns:
        img (np.array): Image with bounding boxes
    '''
    color_map = {0: (0, 255, 0),  # Màu xanh cho lớp 0
                1: (0, 0, 255)}  # Màu đỏ cho lớp 1

    for bbox, score, class_idx in zip(bboxes, scores, classes_name):
        x1, y1, x2, y2 = map(int,bbox)
        
        if class_idx == 0:
            class_name = 'focus'
        elif class_idx == 1:
            class_name = 'not focus'
            
        label = f"{class_name}: {score:.2f}"
        
        color = color_map.get(class_idx, (255, 255, 255))
        
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        
    if is_save:
        saved_img_path = os.path.join(directory_cfg['PREDICT_DIR'], 'output.jpg')
        cv2.imwrite(saved_img_path, img)
        ic(f'Saved at {saved_img_path}')
    return img

def get_train_dicts(img_dir):
    """
    Generates a list of dictionaries containing image and annotation data for training.
    Args:
        img_dir (str): The directory containing the 'images' and 'labels' subdirectories.
    Returns:
        list: A list of dictionaries, each containing:
            - "file_name" (str): The path to the image file.
            - "image_id" (int): A unique identifier for the image.
            - "height" (int): The height of the image.
            - "width" (int): The width of the image.
            - "annotations" (list): A list of annotation dictionaries, each containing:
                - "bbox" (list): Bounding box coordinates [x_min, y_min, x_max, y_max].
                - "bbox_mode" (BoxMode): The mode of the bounding box coordinates.
                - "category_id" (int): The class label of the object.
    """

    dataset_dicts = []
    for idx, filename in enumerate(os.listdir(os.path.join(img_dir, "images"))):
        record = {}
        filepath = os.path.join(img_dir, "images", filename)
        height, width = cv2.imread(filepath).shape[:2]

        record["file_name"] = filepath
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        label_path = os.path.join(img_dir, "labels", filename.replace(".jpg", ".txt"))
        objs = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                data = line.strip().split()
                label = int(data[0])
                label = int(data[0])
                center_x, center_y, bbox_width, bbox_height = map(float, data[1:])

                # Chuyển đổi tọa độ từ YOLO sang Detectron2
                x_min = int((center_x - bbox_width / 2) * width)
                y_min = int((center_y - bbox_height / 2) * height)
                x_max = int((center_x + bbox_width / 2) * width)
                y_max = int((center_y + bbox_height / 2) * height)

                obj = {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": label,
                }
                objs.append(obj)

        record["annotations"] = objs
        dataset_dicts.append(record)

    return dataset_dicts

def register_detectron2_dataset(dataset_name='my_dataset', img_dir=None, type='train'):
    '''
    Register dataset for detectron2
    '''
    dataset_name = 'my_dataset_' + type
    num_classes = detectron2_cfg.MODEL.ROI_HEADS.NUM_CLASSES
    thing_classes = [str(i) for i in range(num_classes)]
    
    # REGISTER:
    DatasetCatalog.register(dataset_name, lambda: get_train_dicts(img_dir))
    MetadataCatalog.get(dataset_name).set(thing_classes=thing_classes)
    ic(dataset_name)
    