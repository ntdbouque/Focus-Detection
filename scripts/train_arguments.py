import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from source.train import detectron_train
import argparse

def get_train_arguments():
    parser = argparse.ArgumentParser(description="Training arguments for my custom training detectron2 model.")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = get_train_arguments()
    detectron_train()
    