# MindSpore YoloV3 Model for MindSpore Challenge 21
All files are in the root folder. 
1. `config.py` contains configuration to edit the model for YoloV3
2. `dataset.py` contains the preprocessing of data into MindRecord, data augmentation operations, and preparing data for YoloV3 training.
3. `utils.py` contains utilies to calculate iou, suppression of overlapping bounding boxes with Non-Maximal Suppression(nms) and calculate precision and recall values.
4. `yolov3.py` contains the model structure.

## Training
Only training on ModelArts are currently supported, you are free to modify the code to enable GPU training, but the operation might not be supported yet.

The file train.py allows a person to train the model, it will download the files(on OBS) using Moxing Framework and processes them.

You need to copy all the files in this directory onto a OBS bucket, place them in a folder called `src`.

> The process of training on ModelArts has been thoroughly demo in [this video](https://www.youtube.com/watch?v=5UhSbU2Kfqg)

## Evaluation Example
[Link](https://github.com/MindSporeChallenge21/msc21_evaluation_example)
