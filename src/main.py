import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import read_frumic as rf

# project tasklist:
# - import template images
# - import image for analysis
# - modify image for clarity
#   - change to monochrome?
# - find contours in the image to locate the script
# - use template matching to get symbols
# - translate symbols into romanization
# - (opt.) translate romanization back into clean symbols to double check.

# ----------------------------------------------------------------------------------------------------------------------------
# input and output reading starts here
# ----------------------------------------------------------------------------------------------------------------------------
TEMPLATE_DIR = "./templates"
OUTPUT_DIR = "./output"
INPUT_DIR = "./input"

# clean output directory
for filename in os.listdir(OUTPUT_DIR):
    file_path = os.path.join(OUTPUT_DIR, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            import shutil
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'Error deleting {file_path}: {e}')

# import input images
input_names = [f[:-4] for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))] 
inputs = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))] 
inputs_img_rgb = [cv.imread(name) for name in inputs]

# perform function on each input image
for i, input_img in enumerate(inputs_img_rgb):
    text, box_img = rf.read_frumic(input_img, TEMPLATE_DIR)

    # write results
    with open(os.path.join(OUTPUT_DIR, input_names[i] + ".txt"), "w") as f:
        f.write(text)
        f.close
    
    write_status = cv.imwrite(os.path.join(OUTPUT_DIR, input_names[i] + ".jpg"), box_img)
    if not write_status:
        print(f"Failed to write image")

    