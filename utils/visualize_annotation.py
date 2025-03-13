'''
Author: Nguyen Truong Duy
Purpose: ultility functions for visualizing annotations.
Updated: 2021/07/15
'''

import cv2
from pybboxes import BoundingBox

def convert_yolo_to_coco(bbox, image_shape):
    '''
    Convert YOLO bounding box to COCO bounding box format.
    Args:
        bbox (list): YOLO bounding box format [x_center, y_center, width, height]
        image_shape (tuple): Image shape (height, width)
    Returns:
        list: COCO bounding box format [x_min, y_min, width, height]
    '''
    x_c, y_c, w, h = bbox
    yolo_box = BoundingBox.from_yolo(x_c, y_c, w, h, image_shape)
    coco_box = yolo_box.to_coco(return_values=True)
    return coco_box


def convert_yolo_to_albumentation(bbox, image_shape):
    '''
    Convert YOLO bounding box to Albumentation bounding box format.
    Args:
        bbox (list): YOLO bounding box format [x_center, y_center, width, height]
        image_shape (tuple): Image shape (height, width)    
    Return: 
        list: Albumentation bounding box format [x_min, y_min, x_max, y_max]
    '''

    x_c, y_c, w, h = bbox
    yolo_box = BoundingBox.from_yolo(x_c, y_c, w, h, image_shape)
    albumentation_box = yolo_box.to_albumentations(return_values=True)
    return albumentation_box

def convert_yolo_to_voc(bbox, image_shape):
    '''
    Convert YOLO bounding box to VOC bounding box format.
    Args:
        bbox (list): YOLO bounding box format [x_center, y_center, width, height]
        image_shape (tuple): Image shape (height, width)
    Returns:
        list: VOC bounding box format [x_min, y_min, x_max, y_max]
    '''
    x_c, y_c, w, h = bbox
    yolo_box = BoundingBox.from_yolo(x_c, y_c, w, h, image_shape)
    voc_box = yolo_box.to_voc(return_values=True)
    return voc_box

def draw_annotations(image, annotations):
    """
    Visualize bounding boxes and their classes on an image.

    Args:
        image (numpy.ndarray): The image on which to draw.
        annotations (list): A list of annotations, where each annotation is a list containing
                            the bounding box coordinates [x1, y1, x2, y2] and the class label.

    Returns:
        numpy.ndarray: The image with the bounding boxes and class labels drawn.
    """
    for annotation in annotations:
        bbox = annotation[0]
        class_label = annotation[1]
        
        # Draw the bounding box
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Put the class label near the bounding box
        cv2.putText(image, class_label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

def read_yolo_annotation(label_path):
    '''
    Read YOLO annotation from a text file.
    Args:
        label_path (str): Path to the YOLO annotation file.
    Returns:
        list: A list of annotations, where each annotation is a list containing the bounding box coordinates [x1, y1, x2, y2] and the class label.
    '''
    annotations = []
    with open(label_path, 'r') as file:
        for line in file:
            line = line.strip().split()
            class_label = line[0]
            bbox = [float(x) for x in line[1:]]
            annotations.append([bbox, class_label])
    return annotations

def main(lst_img_paths, lst_label_paths, output_dir):
    '''
    visualize annotations on images and save to output directory
    Args:
        lst_img_paths (list): List of image paths.
        lst_label_paths (list): List of label paths.
        output_dir (str): Directory to save visualized images
    Returns:
        None
    '''
    
    ic(len(lst_img_paths))
    ic(len(lst_label_paths))
    
    os.makedirs(output_dir, exist_ok=True)
    
    for img_path, label_path in zip(sorted(lst_img_paths), sorted(lst_label_paths)):
        #ic(img_path)
        annotation = read_yolo_annotation(label_path)
        image = cv2.imread(img_path)
        height, width, _ = image.shape
        for i in range(len(annotation)):
            annotation[i][0] = convert_yolo_to_voc(annotation[i][0], (width, height))
        image = draw_annotations(image, annotation)
        
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        cv2.imwrite(output_path, image)

if __name__ == '__main__':
    import os
    import glob
    from icecream import ic
    
    # Visualize preprocess data:
    output_dir = '/workspace/competitions/Sly/detectron2_train_infer/data/vis/data_vis_ver_8'
    data_dir = '/workspace/competitions/Sly/detectron2_train_infer/data/preprocess/data_ver_8'
    
    lst_img_paths = glob.glob(os.path.join(data_dir, 'images', '*.jpg'))
    lst_label_paths = glob.glob(os.path.join(data_dir, 'labels', '*.txt'))
    
    
    ic(len(lst_img_paths))
    ic(len(lst_label_paths))
    
    main(lst_img_paths, lst_label_paths, output_dir)