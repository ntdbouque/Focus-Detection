'''
Author: Nguyen Truong Duy
Purpose: Argument parser for prediction script
Latest Update: 17-02-2025
'''

import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import argparse

from source.predict import detectron_infer, detectron_infer_video

def get_args():
    parser = argparse.ArgumentParser(description="Argument parser for prediction script")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input data')
    parser.add_argument('--output_path', type=str, required=False, help='Path to save the output data')
    parser.add_argument('--is_video', action='store_true', help='Flag to indicate if the input is a video')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    if args.is_video:
        detectron_infer_video(args.checkpoint_path, args.input_path, args.output_path)
    else:
        detectron_infer(args.checkpoint_path, args.input_path)