import cv2

cap = cv2.VideoCapture("gesture-u.avi")
while not cap.isOpened():
        cap = cv2.VideoCapture("gesture-u.avi")
        cv2.waitKey(1000)
        print "Wait for the header"

pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)

cap.set(1, 5)
ret, frame = cap.read()
cv2.imwrite("initial_frame.png", frame)
