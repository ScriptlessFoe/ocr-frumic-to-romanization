import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
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

# clean output
output_dir = "./output"
for filename in os.listdir(output_dir):
    file_path = os.path.join(output_dir, filename)
    try:
        # Check if the item is a file or a symbolic link, then delete
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        # If it's a directory, delete it and all its contents recursively
        elif os.path.isdir(file_path):
            import shutil
            shutil.rmtree(file_path)
    except Exception as e:
        print(f'Error deleting {file_path}: {e}')

# import template and input images
template_dir = "./templates"
template_names = sorted([f[:-4] for f in os.listdir(template_dir) if os.path.isfile(os.path.join(template_dir, f))])
templates = sorted([os.path.join(template_dir, f) for f in os.listdir(template_dir) if os.path.isfile(os.path.join(template_dir, f))])
templates_img = [cv.imread(name, cv.IMREAD_GRAYSCALE) for name in templates]

# resize templates to be small
resized_templates_img = []
for template in templates_img:
    w, h = template.shape[::-1]
    n_w = 20
    aspect_ratio = n_w/w
    n_h = int(h*aspect_ratio)
    resized_templates_img.append(cv.resize(template, (n_w, n_h)))

input_dir = "./input"
input_names = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))] 
inputs = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))] 
inputs_img_rgb = [cv.imread(name) for name in inputs]
inputs_img = [cv.imread(name, cv.IMREAD_GRAYSCALE) for name in inputs]

# do template matching
for i, input_img in enumerate(inputs_img):
    scale_successes = {}
    symbol_locs = []

    for j, template in enumerate(resized_templates_img):
        tw, th = template.shape[::-1]
        # iw, ih = input_img.shape[::-1]

        # # loop over the scales of the image
        # for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        #     # resize image
        #     resized_input = cv.resize(input_img, (int(iw*scale), int(ih*scale)))

        #     if resized_input.shape[0] < th or resized_input.shape[1] < tw:
        #         break

        res = cv.matchTemplate(input_img, template, cv.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where(res >= threshold)
        if (len(loc[0]) != 0):
            # record successful template matches

            # if str(scale) in scale_successes:
            #     scale_successes[str(scale)] += 1
            # else:
            #     scale_successes[str(scale)] = 1
            
            # draw box around identified symbols
            for pt in zip(*loc[::-1]):
                cv.rectangle(inputs_img_rgb[i], pt, (pt[0] + tw, pt[1] + th), (0,0,255), 2)

        symbol_locs.append(loc)
    
    # print image of all detected symbols
    cv.imwrite(os.path.join(output_dir, input_names[i]), inputs_img_rgb[i])

    print(symbol_locs)




        
        
        