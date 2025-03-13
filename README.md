# Detect Behaviour Student

This project aims to detect student behavior to determine focus or unfocus using Detectron2.

## Dataset:
This project using dataset: 

## Installation

### Clone Detectron2

First, clone the Detectron2 repository:

```bash
git clone https://github.com/facebookresearch/detectron2.git
cd detectron2
```

### Set Up Dataset

Prepare your dataset in COCO format. Ensure you have the following structure:

```
dataset/
    ├── train/
    ├── val/

```
Notes: Dataset must be yolov8 format

### Install Requirements

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Model Setup:
All configuration are ready to train, evaluate, infer. For more flexible, please change `config.yaml` in config directory.

## How to Train

To train the model, run the following command:

```bash
python scripts/train_arguments.py 
```

## How to Infer

To perform inference, use the following command:

```bash
python scripts/predict_arguments.py --input_path input.jpg  --checkpoint_path /path/to/weights.pth --is_video
```
Note: `--is_video` argument indicate whether input is video or not 

## How to Eval:
To perform evaluation, use the following command:
Not yet develop
