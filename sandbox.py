from detect import detector
import os
from PIL import Image

input_image = Image.open(os.path.join("data", "samples", "pf-test-0000000039.jpg"))

# boxes = detector(input_image)

boxes = [
    {
        "cls_conf": 0.9970274567604065,
        "cls_pred": 1,
        "x1": 1257,
        "x2": 1363,
        "y1": 648,
        "y2": 773
    },
    {
        "cls_conf": 0.9943934679031372,
        "cls_pred": 1,
        "x1": 545,
        "x2": 661,
        "y1": 280,
        "y2": 406
    },
    {
        "cls_conf": 0.9944857954978943,
        "cls_pred": 1,
        "x1": 1471,
        "x2": 1574,
        "y1": 136,
        "y2": 263
    },
    {
        "cls_conf": 0.9925566911697388,
        "cls_pred": 1,
        "x1": 1441,
        "x2": 1566,
        "y1": 998,
        "y2": 1140
    },
    {
        "cls_conf": 0.995840847492218,
        "cls_pred": 0,
        "x1": 669,
        "x2": 773,
        "y1": 443,
        "y2": 553
    },
    {
        "cls_conf": 0.9865790009498596,
        "cls_pred": 1,
        "x1": 851,
        "x2": 965,
        "y1": 54,
        "y2": 191
    },
    {
        "cls_conf": 0.9958486557006836,
        "cls_pred": 1,
        "x1": 1289,
        "x2": 1419,
        "y1": 12,
        "y2": 133
    },
    {
        "cls_conf": 0.9983610510826111,
        "cls_pred": 0,
        "x1": 768,
        "x2": 875,
        "y1": 437,
        "y2": 552
    },
    {
        "cls_conf": 0.9967150688171387,
        "cls_pred": 0,
        "x1": 459,
        "x2": 561,
        "y1": 396,
        "y2": 504
    },
    {
        "cls_conf": 0.9899147748947144,
        "cls_pred": 1,
        "x1": 1483,
        "x2": 1589,
        "y1": 615,
        "y2": 739
    },
    {
        "cls_conf": 0.9949427247047424,
        "cls_pred": 0,
        "x1": 640,
        "x2": 750,
        "y1": 521,
        "y2": 640
    },
    {
        "cls_conf": 0.9982057809829712,
        "cls_pred": 0,
        "x1": 960,
        "x2": 1070,
        "y1": 878,
        "y2": 983
    },
    {
        "cls_conf": 0.992738664150238,
        "cls_pred": 0,
        "x1": 735,
        "x2": 839,
        "y1": 1004,
        "y2": 1125
    },
    {
        "cls_conf": 0.9971444010734558,
        "cls_pred": 0,
        "x1": 718,
        "x2": 832,
        "y1": 877,
        "y2": 982
    }]

def sorter(item):
    return (item["x1"], item["y1"])

boxes = sorted(boxes, key=sorter)
for box in boxes:
    print(box)
