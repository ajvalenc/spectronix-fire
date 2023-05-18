import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

CONTROL = False
#directory = "/home/ajvalenc/Datasets/spectronix/thermal/fire/flames"
directory = "/home/ajvalenc/Datasets/spectronix/thermal/fire/new/blood_fire_test_02_MicroCalibir_M0000337/"
filenames = sorted(os.listdir(directory))

out_det = cv2.VideoWriter("./output/videos/detect.avi",cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (640, 480))
out_ndet = cv2.VideoWriter("./output/videos/no_detect.avi",cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (640, 480))

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 110

# Filter by Area.
params.filterByArea = True
params.minArea = 1

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.1

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

detector = cv2.SimpleBlobDetector_create(params)
i = 0
detected = 0
lst_max_value = []
lst_min_value = []
while i < len(filenames):
    image_raw = cv2.imread(directory + "/" + filenames[i], cv2.IMREAD_ANYDEPTH)
    image = image_raw.copy()
    max_val = 32124.0
    min_val = 28896.0

    curr_max = np.max(image)
    curr_min = np.min(image)
    lst_max_value.append(curr_max)
    lst_min_value.append(curr_min)

    ret, image = cv2.threshold(image, max_val, max_val, cv2.THRESH_TRUNC)
    image = cv2.bitwise_not(image)
    ret, image = cv2.threshold(image, 2**16 - min_val, 2**16 - min_val, cv2.THRESH_TRUNC)
    image = cv2.bitwise_not(image)
    image = ((image-min_val)/(max_val-min_val)) * 255.
    image = np.uint8(image)
    # image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    #FILTER IMAGE   
    # image = cv2.GaussianBlur(image,(5,5),0)
    keypoints = detector.detect(cv2.bitwise_not(image))

    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints = cv2.putText(im_with_keypoints, filenames[i], (5,15),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,0),1,2)

    # plotting
    cv2.imshow("Fire Detection", im_with_keypoints)
    if (cv2.waitKey(5) > 0): break

    if len(keypoints) != 0:
        out_det.write(im_with_keypoints)
        detected = detected + 1
        if CONTROL:
            cv2.imwrite("./output/fail/control/"+filenames[i], image_raw)
    else:
        out_ndet.write(im_with_keypoints)
        if not CONTROL:
            cv2.imwrite("./output/fail/flames/"+filenames[i], image_raw)

    i = i + 1

out_det.release()
out_ndet.release()
print("Number of images", len(filenames))
print("Number of detections", detected)
print("Detection percentage", 100*detected/len(filenames))

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
