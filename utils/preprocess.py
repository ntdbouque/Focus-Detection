'''
Author: Nguyen Truong Duy
Purpose: ultility functions for preprocessing images and annotations.
Updated: 21-02-2025
'''

import cv2

def check_valid_bounding_box_by_iou(bboxes, overlap_threshold=0.15):
    '''
    Check if bounding boxes of class 0 and class 1 overlap more than a threshold.
    Args:
        bboxes (list): List of bounding boxes with format [class_idx, x_center, y_center, width, height].
        overlap_threshold (float): Threshold for overlap ratio.
    Returns:
        bool: True if no invalid overlap, False otherwise.
    '''
    def compute_iou(box1, box2):
        x1_min = box1[1] - box1[3] / 2
        x1_max = box1[1] + box1[3] / 2
        y1_min = box1[2] - box1[4] / 2
        y1_max = box1[2] + box1[4] / 2

        x2_min = box2[1] - box2[3] / 2
        x2_max = box2[1] + box2[3] / 2
        y2_min = box2[2] - box2[4] / 2
        y2_max = box2[2] + box2[4] / 2

        inter_x_min = max(x1_min, x2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_min = max(y1_min, y2_min)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)

        iou = inter_area / float(box1_area + box2_area - inter_area)
        return iou

    class_0_bboxes = [bbox for bbox in bboxes if bbox[0] == 0]
    class_1_bboxes = [bbox for bbox in bboxes if bbox[0] == 1]

    for box0 in class_0_bboxes:
        for box1 in class_1_bboxes:
            if compute_iou(box0, box1) > overlap_threshold:
                return False
    return True
    
def filter_failed_image(lst_annotation_paths):
    '''
    Filter out image paths where all bounding boxes are invalid.
    
    Args:
        lst_annotation_paths (list): List of annotation paths.
    
    Returns:
        list: List of valid annotation paths.
    '''
    valid_annotation_paths = []
    for annotation_path in lst_annotation_paths:
        with open(annotation_path, 'r') as f:
            lines = f.readlines()
            annotations = []
            for line in lines:
                line = line.split(' ')
                class_idx = int(line[0])
                x_center = float(line[1])
                y_center = float(line[2])
                width = float(line[3])
                height = float(line[4].strip())
                annotation = [class_idx, x_center, y_center, width, height]
                annotations.append(annotation)
            
            #filtered_annotations = check_valid_bounding_box_inside(annotations)
            filtered_annotations = choose_class(annotations, [0,1])
            
            
            if filtered_annotations:
                valid_annotation_paths.append((annotation_path, filtered_annotations))
    
    return valid_annotation_paths

def save_valid_image_and_annotations(valid_annotation_paths, output_dir):
    '''
    Save valid images and their annotations to a new directory based on valid annotation paths.
    
    Args:
        valid_annotation_paths (list): List of valid annotation paths and filtered bounding boxes.
        output_dir (str): Directory to save valid images and annotations.
    
    Returns:
        None
    '''
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(os.path.join(output_dir, 'images')):
        os.makedirs(os.path.join(output_dir, 'images'))
    
    if not os.path.exists(os.path.join(output_dir, 'labels')):
        os.makedirs(os.path.join(output_dir, 'labels'))
    
    for annotation_path, filtered_annotations in valid_annotation_paths:
        image_path = annotation_path.replace('labels', 'images').replace('.txt', '.jpg')
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            output_image_path = os.path.join(output_dir, 'images', os.path.basename(image_path))
            cv2.imwrite(output_image_path, image)
            
            output_annotation_path = os.path.join(output_dir, 'labels', os.path.basename(annotation_path))
            with open(output_annotation_path, 'w') as f_dst:
                for annotation in filtered_annotations:
                    f_dst.write(' '.join(map(str, annotation)) + '\n')
        else:
            print(f'Image not found for annotation: {annotation_path}')


def check_valid_bounding_box_inside(bboxes, tolerance=0.05):
    '''
    Check if bbox class 0 inside bbox belong to class 1 and remove invalid bounding boxes.
    
    Args:
        bboxes (list): List of bounding boxes with format [class_idx, x_center, y_center, width, height].
        
    Returns:
        list: Filtered list of valid bounding boxes.
    '''
    
    class_0_bboxes = []
    class_1_bboxes = []
    valid_bboxes = []
    
    filtered_bboxes = [bbox for bbox in bboxes if bbox[0] in {0, 1}]
    # Separate bounding boxes by class
    for bbox in filtered_bboxes:
        class_idx, x_center, y_center, width, height = bbox
        x_min = x_center - width / 2
        y_min = y_center - height / 2
        x_max = x_center + width / 2
        y_max = y_center + height / 2
        
        if class_idx == 0:
            class_0_bboxes.append([x_min, y_min, x_max, y_max, bbox])
        elif class_idx == 1:
            class_1_bboxes.append([x_min, y_min, x_max, y_max, bbox])
    
    # Filter out invalid class 0 bounding boxes inside class 1 bounding boxes
    for x0_min, y0_min, x0_max, y0_max, bbox in class_0_bboxes:
        inside_any_class_1 = False
        for x1_min, y1_min, x1_max, y1_max, _ in class_1_bboxes:
              if (x0_min - tolerance * (x0_max - x0_min) <= x1_min and
                y0_min - tolerance * (y0_max - y0_min) <= y1_min and
                x0_max + tolerance * (x0_max - x0_min) >= x1_max and
                y0_max + tolerance * (y0_max - y0_min) >= y1_max):
                inside_any_class_1 = True
                break
        if not inside_any_class_1:
            valid_bboxes.append(bbox)
    
    # Add all class 1 bounding boxes since they are always valid
    for _, _, _, _, bbox in class_1_bboxes:
        valid_bboxes.append(bbox)
    
    return valid_bboxes

def choose_class(bboxes, class_idxes):
    '''
    Choose bounding boxes with the specified class index.
    
    Args:
        bboxes (list): List of bounding boxes with format [class_idx, x_center, y_center, width, height].
        class_idx (list[int]): Class indexes to choose.
    
    Returns:
        list: Filtered list of bounding boxes.
    '''
    return [bbox for bbox in bboxes if bbox[0] in class_idxes]

def image_name_preprocess(image_directory, label_directory):
    '''
    Preprocess image names by sorting and renaming them sequentially,
    ensuring corresponding label files are also renamed correctly.
    
    Args:
        image_directory (str): Path to the directory containing images.
        label_directory (str): Path to the directory containing label files.
    
    Returns:
        None
    '''
    image_files = sorted([f for f in os.listdir(image_directory) if f.endswith('.jpg')])
    
    for idx, image_file in enumerate(image_files):
        new_name = f"{idx:05d}.jpg"
        old_image_path = os.path.join(image_directory, image_file)
        new_image_path = os.path.join(image_directory, new_name)
        
        # Rename image file
        os.rename(old_image_path, new_image_path)
        
        # Process corresponding label file
        label_file = os.path.splitext(image_file)[0] + '.txt'  # Assuming label files have .txt extension
        old_label_path = os.path.join(label_directory, label_file)
        new_label_path = os.path.join(label_directory, f"{idx:05d}.txt")
        
        if os.path.exists(old_label_path):  # Ensure the label file exists before renaming
            os.rename(old_label_path, new_label_path)
        else:
            print(f"Warning: No label found for {image_file}")

def main(annotation_dir, output_dir):
    '''
    Filter out invalid bounding boxes and save valid images and annotations to a new directory.
    Args:
        annotation_dir (str): Directory containing annotation files.
        output_dir (str): Directory to save valid images and annotations.
    Returns:
        None
    '''
    os.makedirs(output_dir, exist_ok=True)
    
    lst_annotation_paths = glob.glob(os.path.join(annotation_dir, '*.txt'))
    ic(len(lst_annotation_paths))
    valid_annotation_paths = filter_failed_image(lst_annotation_paths)
    ic(len(valid_annotation_paths))
    save_valid_image_and_annotations(valid_annotation_paths, output_dir)
        
if __name__ == '__main__':
    import os
    import glob
    from icecream import ic
   
    ## Process images and annotations:
    annotation_dir = '/workspace/competitions/Sly/detectron2_train_infer/data/raw/train/labels'
    output_dir = '/workspace/competitions/Sly/detectron2_train_infer/data/preprocess/data_ver_8'
    main(annotation_dir, output_dir)
    
    
    
    # ## Preprocess image names and label names:
    # image_directory = '/workspace/competitions/Sly/detectron2_train_infer/data/raw/valid/images'
    # label_directory = '/workspace/competitions/Sly/detectron2_train_infer/data/raw/valid/labels'
    # image_name_preprocess(image_directory, label_directory)
    