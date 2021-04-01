# import the necessary packages
import numpy as np
import argparse
import cv2



# define argument parsers
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Input image path")
ap.add_argument("-c", "--confidence", type=float, default=0.5, help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())

# define prototext and caffemodel paths
caffeModel = "/home/tokim/code/Face-detection-with-OpenCV-and-deep-learning/models/res10_300x300_ssd_iter_140000.caffemodel"
prototextPath = "/home/tokim/code/Face-detection-with-OpenCV-and-deep-learning/models/deploy.prototxt.txt"

# load our serialized model from disk
print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(prototextPath, caffeModel)


# load the input image and construct an input blob for the image
# by resizing to a fixed 300x300 pixels and then normalizing it
image = cv2.imread(args.get("image"))

# extract the dimensions, resize image into 300x300 and converting image into blobFromImage
(h, w) = image.shape[:2]
# mean subtraction RGB - (104.0, 177.0, 123.0)
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
                             (300, 300), (104.0, 177.0, 123.0))

# pass the blob through the network and obtain the detections and predictions
net.setInput(blob)
detections = net.forward()

# loop over the detections
for i in range(0, detections.shape[2]):
    # extract the confidence (i.e., probability) associated with the prediction
    confidence = detections[0, 0, i, 2]

    # filter detections by confidence greater than the minimum confidence
    # print(confidence)
    if confidence > args["confidence"]:
        # compute the (x, y)-coordinates of the bounding box for the object (multiplication)
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY),
                      (0, 0, 255), 2)
        cv2.putText(image, text, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

print("[INFO] Detection completed")
# resize the image
# image = cv2.resize(image, (1200, 900))
# show the output image
cv2.imshow("Output", image)
cv2.waitKey(0)