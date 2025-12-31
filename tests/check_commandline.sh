#!/bin/bash

##################################################################################
# Checks, if the CLI works
##################################################################################

source .venv/bin/activate

set -e


# predict mmdet
# (Removed mmdet checks as requested)

sahi predict --no_sliced_prediction --model_type yolov5 --source tests/data/coco_utils/terrain1.jpg --novisual --model_path tests/data/models/yolov5/yolov5s6.pt --image_size 320
sahi predict --model_type yolov5 --source tests/data/ --novisual --model_path tests/data/models/yolov5/yolov5s6.pt --image_size 320
sahi predict --model_type yolov5 --source tests/data/coco_utils/terrain1.jpg --export_pickle --export_crop --model_path tests/data/models/yolov5/yolov5s6.pt --image_size 320
sahi predict --model_type yolov5 --source tests/data/coco_utils/ --novisual --dataset_json_path tests/data/coco_utils/combined_coco.json --model_path tests/data/models/yolov5/yolov5s6.pt --image_size 320
# coco yolov5
sahi coco yolov5 --image_dir tests/data/coco_utils/ --dataset_json_path tests/data/coco_utils/combined_coco.json --train_split 0.9
# coco evaluate
sahi coco evaluate --dataset_json_path tests/data/coco_evaluate/dataset.json --result_json_path tests/data/coco_evaluate/result.json
# coco analyse
sahi coco analyse --dataset_json_path tests/data/coco_evaluate/dataset.json --result_json_path tests/data/coco_evaluate/result.json --out_dir tests/data/coco_evaluate/
