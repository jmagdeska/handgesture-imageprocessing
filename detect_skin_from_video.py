from pyimagesearch import imutils
import numpy as np
import os
import argparse
import cv2

letters = ["a", "b", "v", "g", "d", "gj", "e", "zh", "z", "dj", "i", "j", "k", "l", "lj", "m", "n", "nj", "o", "p", "r", "s", "t", "kj", "u", "f", "h", "c", "ch", "dz", "sh"]

def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY, color
    if event == cv2.EVENT_LBUTTONDBLCLK:
        mouseX,mouseY = x,y
        image = cv2.imread(dir_initials + "initial_" + letters[l] + "-1.png")
        initial = imutils.resize(image,width=400, height=300)
        cv2.imwrite("frame.png", initial)
        color = initial[y, x]
        print("My color is " + format(color))
    if event == cv2.EVENT_RBUTTONDBLCLK:
        mouseX,mouseY = x,y
        image = cv2.imread("frame.png")
        image = imutils.resize(image, width=400)
        color_clicked = image[y, x]
        print("Color clicked is " + format(color_clicked))
        if color_clicked[2] < (color[2] + 40) and color_clicked[2] > (color[2] - 40) and color_clicked[1] < (color[1] + 50) and color_clicked[1] > (color[1] - 50) and color_clicked[0] < (color[0] + 50) and color_clicked[0] > (color[0] - 50):
            print("Skin detected")

if not os.path.exists("frames"):
    os.makedirs("frames")
    os.makedirs("frames_initials")
    os.makedirs("frames_skin")

folder = input("Enter which dataset will be processed: ")

dir_videos = "gesture_videos/" + str(folder) + "/"
dir_skin = "frames_skin/" + str(folder) + "/"
dir_frames = "frames/"
dir_initials = "frames_initials/"

if not os.path.exists(dir_skin):
    os.makedirs(dir_skin)

for l in range(31):
    print("Extracting initial frame of letter " + letters[l])
    cap = cv2.VideoCapture(dir_videos + letters[l] + '.avi')
    while not cap.isOpened():
        cap = cv2.VideoCapture(dir_videos + letters[l] + '.avi')
        cv2.waitKey(1000)
        print "Wait for the header"
    k = 1

    cap.set(1, 2)
    ret, frame = cap.read()
    cv2.imwrite(dir_initials + "initial_" + letters[l] + "-1.png", frame)

    for i in range(1,21):
        cap.set(1, i)
        ret, frame = cap.read()
        cv2.imwrite(dir_frames + "initial_" + letters[l] + "-1-" + format(k) + ".png", frame)
        k += 1

for l in range(31):
    print("Select skin color initial frame of letter " + letters[l])
    image_initial = cv2.imread(dir_initials + "initial_" + letters[l] + "-1.png")
    initial = imutils.resize(image_initial,width=400, height=225)
    cv2.namedWindow('images')
    cv2.setMouseCallback('images', draw_circle)
    cv2.imshow("images", initial)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("Separate hand from each frame of letter " + letters[l])
    for i in range(1,21):
        a = cv2.imread(dir_frames + "initial_" + letters[l] + "-1-" + format(i) + ".png")
        initial = imutils.resize(a,width=400, height=225)
        blank_image = np.zeros((300,400,3), np.uint8)
        for a in range(225):
            for b in range(400):
                color_clicked = initial[a,b]
                if color_clicked[2] < (color[2] + 40) and color_clicked[2] > (color[2] - 40) and color_clicked[1] < (color[1] + 60) and color_clicked[1] > (color[1] - 60) and color_clicked[0] < (color[0] + 80) and color_clicked[0] > (color[0] - 80) :
                    blank_image[a,b] = initial[a,b]
        cv2.imwrite(dir_skin + letters[l] + "-1-" + format(i) + ".png", blank_image)

print("Done with processing")