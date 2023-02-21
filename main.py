import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

CONTROL = True
directory = "/home/ajvalenc/Datasets/spectronix/fire/control"
filenames = os.listdir(directory)

out = cv2.VideoWriter("outpy.avi",cv2.VideoWriter_fourcc('M','J','P','G'), 30, (640,480))

# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0;
params.maxThreshold = 211#90;

# Filter by Area.
params.filterByArea = True
params.minArea = 6

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
#for i in range(len(filenames)):
max_value = 0.
min_value = 0.
while i < len(filenames):
    image = cv2.imread(directory + "/" + filenames[i], cv2.IMREAD_ANYDEPTH)
    #threshold and normalize the image and convery to 8-bit for blob detection
    max_val = 32124.0 #30200.0
    min_val = 28896.0 #28160.0

    max_value += np.max(image)
    min_value += np.min(image)

    ret, image = cv2.threshold(image,max_val,max_val,cv2.THRESH_TRUNC)
    image = cv2.bitwise_not(image)
    ret, image = cv2.threshold(image, 2**16 - min_val, 2**16 - min_val, cv2.THRESH_TRUNC)
    image = cv2.bitwise_not(image)
    image = ((image-min_val)/(max_val-min_val)) * 255.
    image = np.uint8(image)
    #image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    #FILTER IMAGE   
    #image = cv2.GaussianBlur(image,(5,5),0)
    keypoints = detector.detect(cv2.bitwise_not(image))

    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im_with_keypoints = cv2.putText(im_with_keypoints, filenames[i], (5,15),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,0,0),1,2)
  
    # plotting
    cv2.imshow("Fire Detection", im_with_keypoints)
    if (cv2.waitKey(5) > 0): break
    out.write(im_with_keypoints)
    
    if len(keypoints) != 0:
        detected = detected + 1
        if CONTROL:
            cv2.imwrite("./fail/control/"+filenames[i], im_with_keypoints)
    else:
        if not CONTROL:
            cv2.imwrite("./fail/flames/"+filenames[i], im_with_keypoints)

    i = i + 1

out.release()
print("Average max", max_value / len(filenames))
print("Average min", min_value / len(filenames))
print("number of flames detected = " + str(detected))
print(detected/len(filenames) * 100)

    
