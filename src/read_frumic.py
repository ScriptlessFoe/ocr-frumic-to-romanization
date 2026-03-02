import cv2 as cv
import numpy as np
import os

# debug
from matplotlib import pyplot as plt

DIST = 6

# import template images
def __import_templates(template_dir):
    template_names = sorted([f[:-4] for f in os.listdir(template_dir) if os.path.isfile(os.path.join(template_dir, f))])
    templates = sorted([os.path.join(template_dir, f) for f in os.listdir(template_dir) if os.path.isfile(os.path.join(template_dir, f))])
    templates_img = [cv.imread(name, cv.IMREAD_GRAYSCALE) for name in templates]

    return template_names, templates_img

# ----------------------------------------------------------------------------------------------------------------------------
# main function def
# ----------------------------------------------------------------------------------------------------------------------------
def read_frumic(input_img_rgb, template_dir):
    template_names, templates_img = __import_templates(template_dir)
    input_img = cv.cvtColor(input_img_rgb, cv.COLOR_BGR2GRAY)

    # determine if text is black on white and correct accordingly:
    mean_value = cv.mean(input_img)
    if (mean_value[0] > 127):
        input_img = cv.bitwise_not(input_img)

    # do template matching, with resizing
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
            # if j == template_names.index("period") and scale == 1.5:
            #     plt.plot,plt.imshow(res,cmap = 'gray', vmax=1)
            #     plt.show()

    # check if successful
    if (len(scale_successes) == 0):
        # Failed, return early
        return "", input_img_rgb

    # get symbol locks with most symbol detections
    scale_max = max(scale_successes, key=scale_successes.get)
    symbol_locs = scale_symbol_locs[scale_max]

    # debug
    # print(scale_max)
    
    # make image of with boxes on all detected symbols
    iw, ih = input_img_rgb.shape[1::-1]
    input_img_rgb = cv.resize(input_img_rgb, (int(iw*scale_max), int(ih*scale_max)))
    for loc in symbol_locs:
        for pt in zip(*loc[::-1]):
            cv.rectangle(input_img_rgb, pt, (pt[0] + 15, pt[1] + 20), (0,0,255), 2)
    input_img_rgb = cv.resize(input_img_rgb, (iw, ih))

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

    # create string, adding spaces
    punctuation_index = [template_names.index("comma"), 
                        template_names.index("comma2"), 
                        template_names.index("period"), 
                        template_names.index("bang"), 
                        template_names.index("amperstand"), 
                        template_names.index("dquote"),
                        template_names.index("lparen"),
                        template_names.index("rparen"),
                        template_names.index("percent"),
                        template_names.index("question"),]
    punctuation_str = [",", ",", ".", "!", "&", "\"", "\'", "(", ")", "%", "?"]
    full_str = ""
    for key in lines:
        line = lines[key]
        line_str = ""
        last = line[0]
        ave_gap = 25
        if (len(line) > 1):
            ave_gap = int(np.mean(np.diff([pt[0][0] for pt in line])))
        for pt in line:
            # smaller symbols get smaller gaps
            gap = ave_gap if (last[1] in [template_names.index("s"), template_names.index("c"), template_names.index("lparen"), template_names.index("dquote")]) else ave_gap + 5
            if abs(last[0][0] - pt[0][0]) >= gap or (last[1] in punctuation_index and last[1] not in [template_names.index("lparen"), template_names.index("dquote")]):
                line_str += " "
            last = pt
            ch = template_names[pt[1]]
            if len(ch) == 1:
                line_str += ch.upper()
            elif pt[1] in punctuation_index:
                line_str += punctuation_str[punctuation_index.index(pt[1])]
            else:
                line_str += "["+ch+"]"
        full_str += line_str + "\n"
        
    return full_str, input_img_rgb