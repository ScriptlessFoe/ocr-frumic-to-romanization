import cv2 as cv
import os

# project tasklist:
# - import template images
# - import image for analysis
# - modify image for clarity
#   - change to monochrome?
# - find contours in the image to locate the script
# - use template matching to get symbols
# - translate symbols into romanization
# - (opt.) translate romanization back into clean symbols to double check.

# import template and input images
template_dir = "../templates/"
templates = [f for f in os.listdir(template_dir) if os.path.isfile(os.path.join(template_dir, f))]
templates_img = []

input_dir = "../input/"
inputs = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))] 

