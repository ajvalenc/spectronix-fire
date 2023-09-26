import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

class_name = ["flames", "control"]
dir_in = "/home/ajvalenc/Datasets/spectronix/thermal/fire/raw"
dir_out = "/home/ajvalenc/Datasets/spectronix/thermal/fire/processed"

lst_max_value = []
lst_min_value = []

max_val = 32124.0 #30866.06
min_val = 28896.0 #28841.40

for folder in class_name:
    dir_in_class = dir_in + "/" + folder
    dir_out_class = dir_out + "/" + folder
    filenames = sorted(os.listdir(dir_in_class))
    i = 0
    while i < len(filenames):
        image_raw = cv2.imread(dir_in_class + "/" + filenames[i], cv2.IMREAD_ANYDEPTH)
        image = image_raw.copy()

        curr_max = np.max(image)
        curr_min = np.min(image)
        lst_max_value.append(curr_max)
        lst_min_value.append(curr_min)

        #ret, image = cv2.threshold(image, max_val, max_val, cv2.THRESH_TRUNC)
        #image = cv2.bitwise_not(image)
        #ret, image = cv2.threshold(image, 2**16 - min_val, 2**16 - min_val, cv2.THRESH_TRUNC)
        #image = cv2.bitwise_not(image)
        image = 255. * ((image-min_val)/(max_val-min_val))
        image = np.clip(image, 0, 255)
        image = np.uint8(image)
        #image = cv2.bitwise_not(image)

        #image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # plotting
        cv2.imshow("Fire Detection", image)
        if (cv2.waitKey(5) > 0): break

        cv2.imwrite(dir_out_class + "/" + filenames[i], image)

        i += 1

# compute stats
max_value = np.array(lst_max_value).max()
min_value = np.array(lst_min_value).min()
avg_max_value = np.array(lst_max_value).mean()
med_max_value = np.median(np.array(lst_max_value))
avg_min_value = np.array(lst_min_value).mean()
med_min_value = np.median(np.array(lst_min_value))

print("Max value", max_value)
print("Min value", min_value)
print("Average max", avg_max_value)
print("Average min", avg_min_value)
print("Median max", med_max_value)
print("Median min", med_min_value)
