# Installation
```
pip install -v -e .
```

**For GPU use, torch version in `requirements.txt` may need to be changed to match CUDA version.

# YOLOX Pretrained Standard Models

|Model |size |mAP<sup>val<br>0.5:0.95 |mAP<sup>test<br>0.5:0.95 | Speed V100<br>(ms) | Params<br>(M) |FLOPs<br>(G)| weights |
| ------        |:---: | :---:    | :---:       |:---:     |:---:  | :---: | :----: |
|[YOLOX-s](./exps/default/yolox_s.py)    |640  |40.5 |40.5      |9.8      |9.0 | 26.8 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth) |
|[YOLOX-m](./exps/default/yolox_m.py)    |640  |46.9 |47.2      |12.3     |25.3 |73.8| [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_m.pth) |
|[YOLOX-l](./exps/default/yolox_l.py)    |640  |49.7 |50.1      |14.5     |54.2| 155.6 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.pth) |
|[YOLOX-x](./exps/default/yolox_x.py)   |640   |51.1 |**51.5**  | 17.3    |99.1 |281.9 | [github](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_x.pth) |

# Training Custom

## Preparation

### Dataset

Dataset needs to be put in the following location
```
./datasets/[dataset_folder_name]
```
The three folders below are required for COCO standard format:
1. annotations
    - train.json
    - valid.json
1. train
1. valid

### Exp file
```
./exps/yolox_coco_custom.py
```

Update training parameters under `__init__()`.

****NOTES**:

1. `self.num_classes` must match the number of classes in `./yolox/data/datasets/coco_classes.py`. Comment out unused classes and add your own.
1. **[EXTREMELY IMPORTANT]**`self.depth` and `self.width` must be set the appropriate values based on the YOLOX model used.
1. Training results, history, and checkpoints will be saved to `self.output_dir`.

#### Other available training parameters

Full list of adjustable training parameters, including transformation configurations, can be found at the following:
```
./yolox/exp/yolox_base.py
```

**DO NOT** edit the `yolox_base.py` file. Add changes to `yolox_coco_custom.py` instead.




### Training start
```
CUDA_LAUNCH_BLOCKING=1 python tools/train.py -f ./exps/yolox_coco_custom.py -d 0 -b 32 --fp16 -o -c ./models_pretrained/yolox_l.pth
```

- -f: exp file
- -d: device (usually 0 is CUDA)
- -b: batch size
- --fp16: mix precision training
- -o: occupy GPU memory first for training
- -c: checkpoint file (yolox pretrained model to use as base)

# Inference

```
python main.py
```

Make sure to set `INFERENCE SETTINGS` at the top of `main.py`.

Explanations to settings are commented inside the file.

****IMPORTANT NOTES**:

1. Do not forget to update `./yolox/data/datasets/coco_classes.py` if using a custom trained model, or reset to default COCO if using pretrained model.
1. Split a single image into smaller grids by setting `grid = True`. Rows and columns must be set to a number that is divisible by the original image height and width respectively.
1. Handle either a directory of images or videos by setting `inference_type`. 
