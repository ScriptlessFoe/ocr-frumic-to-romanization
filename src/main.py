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

DIST = 5
GAP = 25
OUTPUT_DIR = "./output"
TEMPLATE_DIR = "./templates"
INPUT_DIR = "./input"

# clean output directory
for filename in os.listdir(OUTPUT_DIR):
    file_path = os.path.join(OUTPUT_DIR, filename)
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

# import template images
template_names = sorted([f[:-4] for f in os.listdir(TEMPLATE_DIR) if os.path.isfile(os.path.join(TEMPLATE_DIR, f))])
templates = sorted([os.path.join(TEMPLATE_DIR, f) for f in os.listdir(TEMPLATE_DIR) if os.path.isfile(os.path.join(TEMPLATE_DIR, f))])
templates_img = [cv.imread(name, cv.IMREAD_GRAYSCALE) for name in templates]

# import input images
input_names = [f[:-4] for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))] 
inputs = [os.path.join(INPUT_DIR, f) for f in os.listdir(INPUT_DIR) if os.path.isfile(os.path.join(INPUT_DIR, f))] 
inputs_img_rgb = [cv.imread(name) for name in inputs]
inputs_img = [cv.imread(name, cv.IMREAD_GRAYSCALE) for name in inputs]

# do template matching, with resizing
for i, input_img in enumerate(inputs_img):
    scale_successes = {}
    scale_symbol_locs = {}

    for j, template in enumerate(templates_img):
        tw, th = template.shape[::-1]
        iw, ih = input_img.shape[::-1]

        # loop over the scales of the image
        for scale in np.linspace(0.5, 1.5, 10)[::-1]:
            # resize image
            resized_input = cv.resize(input_img, (int(iw*scale), int(ih*scale)))

            if resized_input.shape[0] < th or resized_input.shape[1] < tw:
                break

            res = cv.matchTemplate(resized_input, template, cv.TM_CCOEFF_NORMED)
            threshold = 0.8
            if j == template_names.index("nii"):
                threshold = 0.74
            if j == template_names.index("a"):
                threshold = 0.79
            loc = np.where(res >= threshold)
            if (len(loc[0]) != 0):
                # record successful template matches
                if scale in scale_successes:
                    scale_successes[scale] += 1
                else:
                    scale_successes[scale] = 1
            
            if scale in scale_symbol_locs:
                scale_symbol_locs[scale].append(loc)
            else:
                scale_symbol_locs[scale] = [loc]

            # debug
            # if j == template_names.index("c") and i == input_names.index("messenger") and scale == 1.5:
            #     plt.plot,plt.imshow(res,cmap = 'gray', vmax=1)
            #     plt.show()

    # get symbol locks with most symbol detections
    scale_max = max(scale_successes, key=scale_successes.get)
    symbol_locs = scale_symbol_locs[scale_max]

    # debug
    # if i == input_names.index("messenger"):
    #     print(scale_max)
    
    # make image of with boxes on all detected symbols
    iw, ih = inputs_img_rgb[i].shape[1::-1]
    inputs_img_rgb[i] = cv.resize(inputs_img_rgb[i], (int(iw*scale_max), int(ih*scale_max)))
    for loc in symbol_locs:
        for pt in zip(*loc[::-1]):
            cv.rectangle(inputs_img_rgb[i], pt, (pt[0] + 15, pt[1] + 20), (0,0,255), 2)
    write_status = cv.imwrite(os.path.join(OUTPUT_DIR, input_names[i] + ".jpg"), inputs_img_rgb[i])
    if not write_status:
        print(f"Failed to write image")

    # group points by y value to get lines
    lines = {}
    last = (0,0)
    for j, loc in enumerate(symbol_locs):
        for pt in zip(*loc[::-1]):

            # group near
            found = 0
            for key in lines.keys():
                if abs(key - pt[1]) <= DIST:
                    lines[key].append(((int(pt[0]), int(pt[1])), j))
                    found = 1
                    break
            
            if found == 0:
                lines[pt[1]] = [((int(pt[0]), int(pt[1])), j)]
    
    # sort
    for key in lines:
        lines[key] = sorted(lines[key], key= lambda pt: pt[0][0])
    
    # remove dupes
    for key in lines:
        line = lines[key]
        last = ((-1000,0), 0)
        clean_line = []
        for pt in line:
            if not abs(last[0][0] - pt[0][0]) <= DIST:
                clean_line.append(pt)
            # check edge cases (symbol appears inside another symbol)
            if last[1] != pt[1]:
                if last[1] == template_names.index("d") and pt[1] == template_names.index("uth"):
                    clean_line.pop()
                    clean_line.append(pt)
                if last[1] == template_names.index("k") and pt[1] == template_names.index("uth"):
                    clean_line.pop()
                    clean_line.append(pt)
                if last[1] == template_names.index("s") and pt[1] == template_names.index("n"):
                    clean_line.pop()
                    clean_line.append(pt)
                if last[1] == template_names.index("i") and pt[1] == template_names.index("l"):
                    clean_line.pop()
                    clean_line.append(pt)
                if last[1] == template_names.index("k") and pt[1] == template_names.index("u"):
                    clean_line.pop()
                    clean_line.append(pt)
            last = pt
        lines[key] = clean_line

    # print, adding spaces
    with open(os.path.join(OUTPUT_DIR, input_names[i] + ".txt"), "w") as f:
        for key in lines:
            line = lines[key]
            line_str = ""
            last = line[0]
            for pt in line:
                # smaller symbols get smaller gaps
                gap = (GAP - 5) if (last[1] == template_names.index("s") or last[1] == template_names.index("c")) else GAP
                if abs(last[0][0] - pt[0][0]) >= gap:
                    line_str += " "
                last = pt
                ch = template_names[pt[1]]
                if len(ch) == 1:
                    line_str += ch.upper()
                else:
                    line_str += "["+ch+"]"
            f.write(line_str)
            f.write("\n")
        f.close()
    
    
            
            






        
        
        