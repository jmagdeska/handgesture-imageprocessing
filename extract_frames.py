import cv2

cap = cv2.VideoCapture('C:\Users\Jana\PyCharmProjects\skin-detection/videos/gesture-b-2.avi')
while not cap.isOpened():
        cap = cv2.VideoCapture('C:\Users\Jana\PyCharmProjects\skin-detection/videos/gesture-b-2.avi')
        cv2.waitKey(1000)
        print "Wait for the header"

pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

cap.set(1, 5)
ret, frame = cap.read()
cv2.imwrite("frames/initial_b.png", frame)


