from __future__ import division

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
from torchvision import datasets
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import torch.nn.functional as F

def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad

def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", type=str, default="data/samples", help="path to dataset")
    parser.add_argument("--model_def", type=str, default="config/yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="weights/yolov3.weights", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/coco.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.8, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.4, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
    parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")
    parser.add_argument("--checkpoint_model", type=str, help="path to checkpoint model")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("output", exist_ok=True)

    # Set up model
    model = Darknet("config/yolov3-custom.cfg", img_size=416).to(device)

    if "checkpoints/yolov3_ckpt_2.pth".endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights("checkpoints/yolov3_ckpt_2.pth")
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load("checkpoints/yolov3_ckpt_2.pth", map_location=torch.device('cpu')))

    model.eval()  # Set in evaluation mode

    classes = load_classes("data/custom/classes.names")  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    img = transforms.ToTensor()(Image.open(os.path.join("data", "samples", "pf-test-0000000039.jpg")))
    # Pad to square resolution
    img, _ = pad_to_square(img, 0)
    # Resize
    img = resize(img, 416)
    input_imgs = img

    print("\nPerforming object detection:")
    # Configure input
    input_imgs = Variable(input_imgs.type(Tensor))
    input_imgs = input_imgs.unsqueeze(0)
    
    # Get detections
    with torch.no_grad():
        detections = model(input_imgs)
        detections = non_max_suppression(detections, 0.5, 0.4)

    # Bounding-box colors
    cmap = plt.get_cmap("tab20b")
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]
    
    img = np.array(Image.open(os.path.join("data", "samples", "pf-test-0000000039.jpg")))
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    # Draw bounding boxes and labels of detections
    if detections is not None:
        print(detections)
        # Rescale boxes to original image
        detections = rescale_boxes(detections[0], 416, img.shape[:2])
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = random.sample(colors, n_cls_preds)
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            print("\t+ Label: %s, Conf: %.5f" % (classes[int(cls_pred)], cls_conf.item()))
            box_w = x2 - x1
            box_h = y2 - y1
            #color = bbox_colors[int(np.where(unique_labels == int(cls_pred))[0])]
            color = "red" if int(cls_pred) == 1 else "blue"
            
            # Create a Rectangle patch
            if True:
                bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor=color, facecolor="none")
                # Add the bbox to the plot
                ax.add_patch(bbox)
                # Add label
                # plt.text(x1,y1,s=classes[int(cls_pred)],color="white",verticalalignment="top",bbox={"color": color, "pad": 0},)
            
            # img_crop = tmp_img.crop((int(x1),int(y1),int(x1) + int(box_w),int(y1) + int(box_h)))
            # img_crop.save(os.path.join("bbox_output",img_name + "_{:010d}.jpg".format(count)),"JPEG")
            # print("{} {} {} {}".format(x1,y1,box_w,box_h))
#                count += 1                
    # Save generated image with detections
    plt.axis("off")
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    plt.savefig(os.path.join("output", "pf-test-0000000039.jpg"), bbox_inches="tight", pad_inches=0.0)
    plt.close()

        
