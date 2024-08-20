import os
import time
import glob
import copy
import json
import csv

import numpy as np
import cv2
import torch

from predictor import Predictor
from yolox.exp.build import get_exp



# ----------------------- INFERENCE SETTINGS ----------------------------
## DEFAULT PRETRAINED MODELS -- comment out if using custom
## if using default pretrained models, choose from the following:
# yolo_model = 'yolox-s'
# yolo_model = 'yolox-m'
yolo_model = 'yolox-l'
# yolo_model = 'yolox-x'

## CUSTOM TRAINED MODELS  -- comment out if using default pretrained
# yolo_model = './exps/yolox_coco_custom.py' # must refer to exp file path
## custom trained ckpt  -- not used if using default pretrained
custom_ckpt = 'best_ckpt.pth' # place ckpt in models_custom/ 

## set grid inference settings - detect by splitting image into grid
grid = False
rows = 4 # must choose number that is divisible by image height
cols = 4 # must choose number that is divisible by image width

## inference settings
test_conf = 0.6 # detection confidence threshold
nmsthre = 0.45 # detection bbox overlap threshold
test_size = (640, 640) # default training size is (640, 640)

## input folder
input_folder = './input/human'
inference_type = 'video' # 'image' or 'video'





# ---------------------- START INFERENCE ---------------------------------
default_pretrained = [
    'yolox-s',
    'yolox-m',
    'yolox-l',
    'yolox-x'
]

if yolo_model in default_pretrained:
    exp = get_exp(exp_name=yolo_model)
else:
    exp = get_exp(exp_file=yolo_model)

# adjust inference settings
exp.test_conf = test_conf
exp.nmsthre = nmsthre
exp.test_size = test_size

model = exp.get_model()
if torch.cuda.is_available():
    device = 'cuda'
    model.cuda()
else:
    device = 'cpu'
model.eval()

if yolo_model in default_pretrained:
    ckpt = torch.load(f"models_pretrained/{yolo_model.replace('-','_')}.pth", map_location="cpu")
else:
    ckpt = torch.load(f"models_custom/{custom_ckpt}", map_location="cpu")
    yolo_model = 'custom'
model.load_state_dict(ckpt['model'])

predictor = Predictor(
    model=model,
    exp=exp,
    device=device, 
)

current_time = time.localtime()
vis_folder = f'output/{yolo_model}/{time.strftime("%Y_%m_%d_%H_%M_%S", current_time)}_conf-{exp.test_conf}_grid-{grid}'
os.makedirs(vis_folder, exist_ok=True)

with open(vis_folder + '/detect_times.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['file', 'model', 'resolution', 'grids', 'grid_size', 'avg_time_per_frame'])

X_test = glob.glob(f"{input_folder}/*")
X_test = sorted(X_test)
    
bbox_data = {}
detect_time = []

if inference_type == 'image':
    for i in X_test:
        bbox_data[os.path.basename(i)] = []
        
        img = cv2.imread(i, cv2.IMREAD_UNCHANGED)
        imgheight = img.shape[0]
        imgwidth = img.shape[1]
        
        if grid:      
            y1 = 0
            M = imgheight // rows
            N = imgwidth // cols

            img_final = copy.deepcopy(img)
            # img_test = copy.deepcopy(img)
            for y in range(0, imgheight, M):
                for x in range(0, imgwidth, N):
                    y1 = y + M
                    x1 = x + N
                    tiles = img[y:y+M, x:x+N]
                    
                    start = time.time()
                    outputs, img_info = predictor.inference(tiles)
                    detect_time.append(time.time()-start)
                    result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
                    
                    img_final[y:y+M,x:x+N] = result_image
                    
                    if outputs[0] is not None:
                        bboxes = outputs[0].cpu()[:, 0:4]
                        bboxes /= img_info["ratio"]
                        bboxes = [(bbox + np.array([x, y, x, y])).tolist() for bbox in bboxes]
                        # for bbox in bboxes:
                        #     x0 = int(bbox[0])
                        #     y0 = int(bbox[1])
                        #     x1 = int(bbox[2])
                        #     y1 = int(bbox[3])
                        #     cv2.rectangle(img_test, (x0, y0), (x1, y1), (0, 0, 255), 2)
                        bbox_data[os.path.basename(i)].extend(bboxes)
                        # cv2.imwrite('bbox_test.jpg', img_test)
        else:
            start = time.time()
            outputs, img_info = predictor.inference(img)
            detect_time.append(time.time()-start)
            img_final = predictor.visual(outputs[0], img_info, predictor.confthre)
            if outputs[0] is not None:
                bboxes = outputs[0].cpu()[:, 0:4]
                bboxes /= img_info["ratio"]
                bbox_data[os.path.basename(i)].extend(bboxes.tolist())    
        
        with open(vis_folder + '/detect_times.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                os.path.basename(i),
                yolo_model,
                f'{imgwidth}x{imgheight}',
                f'{cols if grid else 1}x{rows if grid else 1}', 
                f'{int(imgwidth/cols if grid else imgwidth)}x{int(imgheight/rows if grid else imgheight)}', 
                np.mean(detect_time)
                ])
        
        
        save_file_name = os.path.join(vis_folder, os.path.basename(i))
        print("Saving detection result in {}".format(save_file_name))
        cv2.imwrite(save_file_name, img_final)
        
                
    with open(vis_folder + "/bbox.json", "w") as f:
        json.dump(bbox_data, f, indent=4)

else:
    for i in X_test:
        # bbox_data[os.path.basename(i)] = []
        
        cap = cv2.VideoCapture(i)
        fps = round(cap.get(cv2.CAP_PROP_FPS))
        imgwidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        imgheight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"frames: {frame_count}")
        
        writer = cv2.VideoWriter(os.path.join(vis_folder, os.path.basename(i)), cv2.VideoWriter_fourcc(*"mp4v"), fps, (imgwidth, imgheight))
        
        while True:
            ret, frame = cap.read()
            if ret:
                img = frame
                if grid:      
                    y1 = 0
                    M = imgheight // rows
                    N = imgwidth // cols

                    img_final = copy.deepcopy(img)
                    # img_test = copy.deepcopy(img)
                    for y in range(0, imgheight, M):
                        for x in range(0, imgwidth, N):
                            y1 = y + M
                            x1 = x + N
                            tiles = img[y:y+M, x:x+N]
                            
                            start = time.time()
                            outputs, img_info = predictor.inference(tiles)
                            detect_time.append(time.time()-start)
                            result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
                            
                            img_final[y:y+M,x:x+N] = result_image
                            
                            # if outputs[0] is not None:
                            #     bboxes = outputs[0].cpu()[:, 0:4]
                            #     bboxes /= img_info["ratio"]
                            #     bboxes = [(bbox + np.array([x, y, x, y])).tolist() for bbox in bboxes]
                            #     # for bbox in bboxes:
                            #     #     x0 = int(bbox[0])
                            #     #     y0 = int(bbox[1])
                            #     #     x1 = int(bbox[2])
                            #     #     y1 = int(bbox[3])
                            #     #     cv2.rectangle(img_test, (x0, y0), (x1, y1), (0, 0, 255), 2)
                            #     bbox_data[os.path.basename(i)].extend(bboxes)
                            #     # cv2.imwrite('bbox_test.jpg', img_test)
                else:
                    start = time.time()
                    outputs, img_info = predictor.inference(img)
                    detect_time.append(time.time()-start)
                    img_final = predictor.visual(outputs[0], img_info, predictor.confthre)
                    # if outputs[0] is not None:
                    #     bboxes = outputs[0].cpu()[:, 0:4]
                    #     bboxes /= img_info["ratio"]
                    #     bbox_data[os.path.basename(i)].extend(bboxes)
                
                writer.write(img_final)
                
            else:
                break    
        
        writer.release()
        
        with open(vis_folder + '/detect_times.csv', 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                os.path.basename(i),
                yolo_model,
                f'{imgwidth}x{imgheight}',
                f'{cols if grid else 1}x{rows if grid else 1}', 
                f'{int(imgwidth/cols if grid else imgwidth)}x{int(imgheight/rows if grid else imgheight)}', 
                np.mean(detect_time)
                ])        
        