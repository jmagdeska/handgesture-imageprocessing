# USAGE
# python skindetector.py
# python skindetector.py --video video/skin_example.mov

# import the necessary packages
from pyimagesearch import imutils
import numpy as np
import argparse
import cv2

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY, color
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX,mouseY = x,y
        cv2.imwrite("screen.png", frame)
        image = cv2.imread("screen.png")
        color = image[x, y]
        print("My color is " + format(color))
    if event == cv2.EVENT_RBUTTONDBLCLK:
        mouseX,mouseY = x,y
        cv2.imwrite("screen2.png", frame)
        image = cv2.imread("screen2.png")
        color_clicked = image[x, y]
        print("Color clicked is " + format(color_clicked))
        if color_clicked[2] < (color[2] + 40) and color_clicked[2] > (color[2] - 40) and color_clicked[1] < (color[1] + 50) and color_clicked[1] > (color[1] - 50) and color_clicked[0] < (color[0] + 80) and color_clicked[0] > (color[0] - 80):
            print("Skin detected")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
args = vars(ap.parse_args())

# define the upper and lower boundaries of the HSV pixel
# intensities to be considered 'skin'
lower = np.array([0, 48, 80], dtype="uint8")
upper = np.array([20, 255, 255], dtype="uint8")

# if a video path was not supplied, grab the reference
# to the gray
if not args.get("video", False):
    camera = cv2.VideoCapture(0)

# otherwise, load the video
else:
    camera = cv2.VideoCapture(args["video"])

# keep looping over the frames in the video
list = []

while True:
    # grab the current frame
    (grabbed, frame) = camera.read()

    # if we are viewing a video and we did not grab a
    # frame, then we have reached the end of the video
    if args.get("video") and not grabbed:
        break

    # resize the frame, convert it to the HSV color space,
    # and determine the HSV pixel intensities that fall into
    # the speicifed upper and lower boundaries
    frame = imutils.resize(frame, width=400)
    converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, lower, upper)

    # apply a series of erosions and dilations to the mask
    # using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skinMask = cv2.erode(skinMask, kernel, iterations=2)
    skinMask = cv2.dilate(skinMask, kernel, iterations=2)

    # blur the mask to help remove noise, then apply the
    # mask to the frame
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(frame, frame, mask=skinMask)
    cv2.namedWindow('images')
    cv2.setMouseCallback('images', draw_circle)

    # show the skin in the image along with the mask
    cv2.imshow("images", frame)

    # if the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    elif cv2.waitKey(1) & 0xFF == ord("c"):
        cv2.imwrite("test_data/test" + format(len(list) + 1) + ".png", frame)
        list.append("test" + format(len(list) + 1) + ".png")
        print("Screenshot number " + format(len(list))+ " saved !")

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()

