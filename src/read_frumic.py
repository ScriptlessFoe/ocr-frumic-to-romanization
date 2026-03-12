import cv2 as cv
from cv2.typing import MatLike
import numpy as np
import os

# PARAMETERS
DIST = 6
START_RESIZE = 0.5
END_RESIZE = 1.5
NUM_OF_RESIZE_STEPS = 10
THRESHOLD = 0.8
ADJUSTED_THRESHOLDS = {"nii": 0.74, "a": 0.79}
PUNCTUATION_TEMPLATE_NAMES = ["comma", "comma2", "period", "bang", "amperstand", "dquote", "lparen", "rparen", "percent", "question"]
PUNCTUATION_STR = [",", ",", ".", "!", "&", "\"", "(", ")", "%", "?"]
TRANS_STR = {"c": "sh", "d": "th", "nii": "nif", "az": "as", "ek": "eksh"}
SYMBOL_OVERLAPS = {"d": ["uth"], "k": ["uth", "u"], "s": ["n"], "i": ["l"]}

# import template images
def __import_templates(template_dir: str) -> tuple[list[str], list[MatLike]]:
    template_names = sorted([f[:-4] for f in os.listdir(template_dir) if os.path.isfile(os.path.join(template_dir, f))])
    templates = sorted([os.path.join(template_dir, f) for f in os.listdir(template_dir) if os.path.isfile(os.path.join(template_dir, f))])
    templates_img = [cv.imread(name, cv.IMREAD_GRAYSCALE) for name in templates]

    return template_names, templates_img

# do template matching, with resizing
def __resizable_template_matching(template_names: list[str], templates_img: list[MatLike], input_img: MatLike) -> tuple[float, list[tuple[np.ndarray, np.ndarray]]]:
    scale_successes: dict[float, int] = {}
    scale_symbol_locs: dict[float, list[tuple[np.ndarray, np.ndarray]]] = {}
    for j, template in enumerate(templates_img):
        tw, th = template.shape[::-1]
        iw, ih = input_img.shape[::-1]

        # loop over the scales of the image
        for scale in np.linspace(START_RESIZE, END_RESIZE, NUM_OF_RESIZE_STEPS)[::-1]:
            # resize image
            resized_input = cv.resize(input_img, (int(iw*scale), int(ih*scale)))

            if resized_input.shape[0] < th or resized_input.shape[1] < tw:
                break

            res = cv.matchTemplate(resized_input, template, cv.TM_CCOEFF_NORMED)
            threshold = THRESHOLD
            if template_names[j] in ADJUSTED_THRESHOLDS:
                threshold = ADJUSTED_THRESHOLDS[template_names[j]]
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

    # check if successful
    if (len(scale_successes) == 0):
        # Failed, return early
        return None

    # get symbol locks with most symbol detections
    scale_max = max(scale_successes, key=scale_successes.get)
    symbol_locs = scale_symbol_locs[scale_max]

    return scale_max, symbol_locs

# create detection boxes on identified symbols in input image
def __create_detection_image(input_img_rgb: MatLike, scale_max: float, symbol_locs: list[tuple[np.ndarray, np.ndarray]]) -> MatLike:
    iw, ih = input_img_rgb.shape[1::-1]
    input_img_rgb = cv.resize(input_img_rgb, (int(iw*scale_max), int(ih*scale_max)))
    for loc in symbol_locs:
        for pt in zip(*loc[::-1]):
            cv.rectangle(input_img_rgb, pt, (pt[0] + 15, pt[1] + 20), (0,0,255), 2)
    input_img_rgb = cv.resize(input_img_rgb, (iw, ih))

    return input_img_rgb

# comparison helper function for symbols within symbols
def __overlap_compare(last: int, cur: int, template_names: list[str]) -> bool:
    if template_names[last] in SYMBOL_OVERLAPS:
        if template_names[cur] in SYMBOL_OVERLAPS[template_names[last]]:
            return True
    return False

# clean and sort points by line, in left to right order
def __clean_loc_points(template_names: list[str], symbol_locs: list[tuple[np.ndarray, np.ndarray]]) -> dict[int, tuple[tuple[int, int], int]]:
    # group points by y value to get lines
    lines: dict[int, tuple[tuple[int, int], int]] = {}
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
                lines[int(pt[1])] = [((int(pt[0]), int(pt[1])), j)]
    
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
                if __overlap_compare(last[1], pt[1], template_names):
                    clean_line.pop()
                    clean_line.append(pt)
            last = pt
        lines[key] = clean_line
    
    return lines

# create encoding and transcription strings
def __create_messages(template_names: list[str], lines: dict[int, tuple[tuple[int, int], int]]) -> tuple[str, str]:
    punctuation_index = [template_names.index(name) for name in PUNCTUATION_TEMPLATE_NAMES]
    encoded_str = ""
    transcribed_str = ""
    for key in lines:
        line = lines[key]
        line_str = ""
        trans_line_str = ""
        last = line[0]
        ave_gap = 25
        if (len(line) > 1):
            ave_gap = int(np.mean(np.diff([pt[0][0] for pt in line])))
        for pt in line:
            # smaller symbols get smaller gaps
            gap = ave_gap if (last[1] in [template_names.index("s"), template_names.index("c"), template_names.index("lparen"), template_names.index("dquote")]) else ave_gap + 5
            if abs(last[0][0] - pt[0][0]) >= gap or (last[1] in punctuation_index and last[1] not in [template_names.index("lparen"), template_names.index("dquote")]):
                line_str += " "
                trans_line_str += " "
            last = pt
            ch = template_names[pt[1]]
            if len(ch) == 1:
                line_str += ch.upper()
                if ch in TRANS_STR:
                    trans_line_str += TRANS_STR[ch]
                else:
                    trans_line_str += ch
            elif pt[1] in punctuation_index:
                line_str += PUNCTUATION_STR[punctuation_index.index(pt[1])]
                trans_line_str += PUNCTUATION_STR[punctuation_index.index(pt[1])]
            else:
                line_str += "["+ch+"]"
                if ch in TRANS_STR:
                    trans_line_str += TRANS_STR[ch]
                else:
                    trans_line_str += ch
        encoded_str += line_str + "\n"
        transcribed_str += trans_line_str + "\n"
    
    return encoded_str, transcribed_str

# ----------------------------------------------------------------------------------------------------------------------------
# main function def
# ----------------------------------------------------------------------------------------------------------------------------
def read_frumic(input_img_rgb: MatLike, template_dir: str) -> tuple[str, str, MatLike]:
    template_names, templates_img = __import_templates(template_dir)
    input_img = cv.cvtColor(input_img_rgb, cv.COLOR_BGR2GRAY)

    # determine if text is black on white and correct accordingly:
    mean_value = cv.mean(input_img)
    if (mean_value[0] > 127):
        input_img = cv.bitwise_not(input_img)

    # perform template matching
    scale_max, symbol_locs = __resizable_template_matching(template_names, templates_img, input_img)

    # clean up data
    lines = __clean_loc_points(template_names, symbol_locs)

    # create output strings
    encoded_str, transcribed_str = __create_messages(template_names, lines)
    
    # make image of with boxes on all detected symbols
    input_img_rgb = __create_detection_image(input_img_rgb, scale_max, symbol_locs)
        
    return encoded_str, transcribed_str, input_img_rgb