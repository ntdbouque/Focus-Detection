'''
Author: Nguyen Truong Duy
Purpose: Post-processing the inference result
Latest Update: 18-02-2025
'''

def class_preprocess(pred_classes):
    '''
    Preprocess the predicted classes
    Args:
        pred_classes (np.ndarray): Array of predicted classes
    Returns:
        classes_name (list): List of class names
    '''
    classes_name = []
    for class_id in pred_classes:
        if class_id == 0:
            classes_name.append('Focus')
        elif class_id == 1:
            classes_name.append('Unfocus')
        elif class_id == 2:
            classes_name.append('Unfocus')
    return classes_name