import numpy as np
import cv2

IMGS_BASE_DIR='G:/data/ultrasound-nerve-segmentation'
ultrasound_bp_cascade_path=IMGS_BASE_DIR+'/opencv/data/cascade.xml'
ultrasound_bp_test=IMGS_BASE_DIR+'/tests/1_9.tif'
ultrasound_bp_cascade = cv2.CascadeClassifier(ultrasound_bp_cascade_path)

img = cv2.imread(ultrasound_bp_test)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
nerve = ultrasound_bp_cascade.detectMultiScale(
    gray,
    scaleFactor=50,
    minNeighbors=50,
    minSize=(70, 70),
    maxSize=(90, 90),
    flags = cv2.CASCADE_SCALE_IMAGE
)
print nerve[0]#, nerve[1], nerve[2]
for (x,y,w,h) in nerve:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    # roi_gray = gray[y:y+h, x:x+w]
    # roi_color = img[y:y+h, x:x+w]
    # eyes = eye_cascade.detectMultiScale(roi_gray)
    # for (ex,ey,ew,eh) in eyes:
        # cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# opencv_createsamples -info positives.dat -num 1000 -vec positives.vec -w 24 -h 24 580 x 420