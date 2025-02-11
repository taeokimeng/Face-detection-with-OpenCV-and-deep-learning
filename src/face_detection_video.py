# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
import os



# define prototext and caffemodel paths
# Path should be absolute path
# caffeModel = "/models/res10_300x300_ssd_iter_140000.caffemodel"
# prototextPath = "/models/deploy.prototxt.txt"
caffemodel_abs_path = os.path.abspath(__file__ + "/../../models/res10_300x300_ssd_iter_140000.caffemodel")
prototxt_abs_path = os.path.abspath(__file__ + "/../../models/deploy.prototxt.txt")

# load our serialized model from disk
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt_abs_path, caffemodel_abs_path)

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
# if you use Raspberry Pi + picamera
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # extract the dimensions, resize image into 300x300 and converting image into blobFromImage
    (h, w) = frame.shape[:2]
    # mean subtraction RGB - (104.0, 177.0, 123.0)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the detections and predictions
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # filter detections by confidence greater than the minimum confidence
        if confidence < 0.5:
            continue

        # compute the (x, y)-coordinates of the bounding box for the object (multiplication)
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        print(confidence)

        # draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    # show the output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
