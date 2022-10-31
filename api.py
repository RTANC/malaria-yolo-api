from detect import detector
import os
from PIL import Image

input_image = Image.open(os.path.join("data", "samples", "pf-test-0000000039.jpg"))

boxes = detector(input_image)
print(boxes)
