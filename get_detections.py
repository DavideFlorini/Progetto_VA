from __future__ import division

import sys
sys.path.insert(0, 'YOLOv3')

from models import *
from utils.utils import *
from utils.datasets import *

import os
import sys
import time
import datetime
import argparse

from PIL import Image

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import  cv2



parser = argparse.ArgumentParser()
parser.add_argument("--video_path", type=str, help="path to the video")
parser.add_argument("--num_frame", type=int, default=-1 , help="path to the video")

#parser.add_argument("--image_folder", type=str, default="YOLOv3/data/samples", help="path to dataset")
parser.add_argument("--model_def", type=str, default="YOLOv3/config/yolov3.cfg", help="path to model definition file")
parser.add_argument("--weights_path", type=str, default="YOLOv3/weights/yolov3.weights", help="path to weights file")
parser.add_argument("--class_path", type=str, default="YOLOv3/data/coco.names", help="path to class label file")
parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
opt = parser.parse_args()
print(opt)


index1 = max(opt.video_path.rfind('\\'), opt.video_path.rfind('/'))+1
index2 = opt.video_path.rfind('.')

vid_name=opt.video_path[index1:index2]
print(vid_name)

cap = cv2.VideoCapture(opt.video_path)
if (cap.isOpened() == False):
    print("Error opening video stream or file")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("detector_output", exist_ok=True)

# Set up model
model = Darknet(opt.model_def, img_size=opt.img_size).to(device)

if opt.weights_path.endswith(".weights"):
    # Load darknet weights
    model.load_darknet_weights(opt.weights_path)
else:
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path))

model.eval()  # Set in evaluation mode


classes = load_classes(opt.class_path)  # Extracts class labels from file

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
if not os.path.exists('Detector_outputs'):
    os.makedirs('Detector_outputs')

with open('Detector_outputs\\'+vid_name+'_detections.txt', mode='w', encoding='utf-8', buffering=1) as det:
    print("\nPerforming object detection:")
    prev_time = time.time()
    # for frame_i, (img_paths, input_imgs) in enumerate(dataloader):
        # Configure input
    frame_i=0
    if opt.num_frame < 0:
        lim = frame_i + 1
    else:
        lim = opt.num_frame

    while (cap.isOpened() and frame_i<lim):

        ret, frame = cap.read()
        if ret==True:
            f = 600 / max(frame.shape)
            frame=cv2.resize(frame, dsize=(0, 0), fx=f, fy=f)
            if ret == True:
                frame_i=frame_i+1
                if opt.num_frame < 0:
                    lim = frame_i + 1

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pilimg = Image.fromarray(frame)
                ratio = min(opt.img_size / pilimg.size[0], opt.img_size / pilimg.size[1])
                imw = round(pilimg.size[0] * ratio)
                imh = round(pilimg.size[1] * ratio)
                img_transforms = transforms.Compose([transforms.Resize((imh, imw)),
                                                     transforms.Pad((max(int((imh - imw) / 2), 0),
                                                                     max(int((imw - imh) / 2), 0),
                                                                     max(int((imh - imw) / 2), 0),
                                                                     max(int((imw - imh) / 2), 0)), (128, 128, 128)),
                                                     transforms.ToTensor(),
                                                     ])
                image_tensor = img_transforms(pilimg).float()
                image_tensor = image_tensor.unsqueeze_(0)
                input_img = Variable(image_tensor.type(Tensor))

                # Get detections
                with torch.no_grad():
                    detections = model(input_img)
                    detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

                # Log progress
                current_time = time.time()
                inference_time = datetime.timedelta(seconds=current_time - prev_time)
                prev_time = current_time
                print("\t+ Frame %d, Inference Time: %s" % (frame_i, inference_time))

                if detections[0] is not None:
                    # Rescale boxes to original image
                    detections = rescale_boxes(detections[0], opt.img_size, frame.shape[:2])
                    unique_labels = detections[:, -1].cpu().unique()
                    n_cls_preds = len(unique_labels)
                    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

                        print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))

                        box_w = x2 - x1
                        box_h = y2 - y1

                        row=[frame_i, int(cls_pred), float(x1), float(y1), float(box_w), float(box_h), float(conf)]

                        det.write('%d, %d, %f, %f, %f, %f, %f\n'%(frame_i, int(cls_pred), float(x1), float(y1), float(box_w), float(box_h), float(conf)))
        else:
            cap.release()

