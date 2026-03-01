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

output_dir = "./output/"
debug_dir = "./debug/"

# import template and input images
template_dir = "./templates/"
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

input_dir = "./input/"
input_names = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))] 
inputs = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))] 
inputs_img_rgb = [cv.imread(name) for name in inputs]
inputs_img = [cv.imread(name, cv.IMREAD_GRAYSCALE) for name in inputs]

#turn images into canny edges
# canny_templates = []
# for template in resized_templates_img:
#     canny_templates.append(cv.Canny(template,100,200))

# canny_inputs = []
# for input in inputs_img:
#     canny_inputs.append(cv.Canny(input,100,200))

#     plt.subplot(121),plt.imshow(input,cmap = 'gray')
#     plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(cv.Canny(input,100,200),cmap = 'gray')
#     plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    
#     plt.show()

# do template matching
for i, input_img in enumerate(inputs_img):
    for j, template in enumerate(resized_templates_img):
        tw, th = template.shape[::-1]
        iw, ih = input_img.shape[::-1]

        # loop over the scales of the image
        # for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        #     # resize image
        #     resized_input = cv.resize(input_img, (int(iw*scale), int(ih*scale)))

        #     if resized_input.shape[0] < th or resized_input.shape[1] < tw:
        #         break

        res = cv.matchTemplate(input_img, template, cv.TM_CCOEFF_NORMED)
        threshold = 0.6
        loc = np.where( res >= threshold)
        img_cpy = inputs_img_rgb[i].copy()
        for pt in zip(*loc[::-1]):
            cv.rectangle(img_cpy, pt, (pt[0] + tw, pt[1] + th), (0,0,255), 2)
        
        cv.imwrite(os.path.join(output_dir, template_names[j]+"_"+input_names[i]), img_cpy)
        
        
        