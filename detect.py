from __future__ import division

from models import *
from utils.utils import *
from utils.datasets import *

import torch
from torch.autograd import Variable
import torch.nn.functional as F

import io, base64
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from PIL import Image, ImageDraw as Drawer

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

def sorter(val):
    return (round(val[0].item()), round(val[1].item()))

def detector(input_image):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up model
    model = Darknet("config/yolov3-custom.cfg", img_size=416).to(device)

    if "checkpoints/yolov3_ckpt_2.pth".endswith(".weights"):
        # Load darknet weights
        model.load_darknet_weights("checkpoints/yolov3_ckpt_2.pth")
    else:
        # Load checkpoint weights
        model.load_state_dict(torch.load("checkpoints/yolov3_ckpt_2.pth", map_location=torch.device('cpu')))

    model.eval()  # Set in evaluation mode

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
    # read input_image
    # input_image = Image.open(os.path.join("data", "samples", "pf-test-0000000039.jpg"))

    img = transforms.ToTensor()(input_image)
    # Pad to square resolution
    img, _ = pad_to_square(img, 0)
    # Resize
    img = resize(img, 416)

    print("\nPerforming object detection:")
    # Configure input
    img = Variable(img.type(Tensor))
    img = img.unsqueeze(0)
    
    # Get detections
    with torch.no_grad():
        detections = model(img)
        detections = non_max_suppression(detections, 0.5, 0.4)
    
    img = np.array(input_image)

    # Draw bounding boxes and labels of detections
    pred_boxes = []

    if detections is not None:
        # Rescale boxes to original image
        detections = rescale_boxes(detections[0], 416, img.shape[:2])
        # print(len(detections))
        detections = sorted(detections, key=sorter)
        count = 1

        draw = Drawer.Draw(input_image)
        
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:

            color = "red" if int(cls_pred) == 1 else "blue"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

            pred_boxes.append({
                "id": "{:03d}".format(count),
                "x1": round(x1.item()),
                "y1": round(y1.item()),
                "x2": round(x2.item()),
                "y2": round(y2.item()),
                "cls_pred": int(cls_pred),
                "cls_conf": cls_conf.item()
            })
            count += 1
        
        img_buf = io.BytesIO()
        input_image.save(img_buf, format='JPEG')
        img_str = base64.b64encode(img_buf.getvalue())
        img_str = 'data:image/jpg;base64,' +  (str(img_str)[:-1]).replace("b'", "")
        img_buf.close()

    return pred_boxes, img_str