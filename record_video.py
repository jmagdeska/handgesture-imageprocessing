import numpy as np
import cv2

cap = cv2.VideoCapture(0)

letters = ["a", "b", "v", "g", "d", "gj", "e", "zh", "z", "dj", "i", "j", "k", "l", "lj", "m", "n", "nj", "o", "p", "r", "s", "t", "kj", "u", "f", "h", "c", "ch", "dz", "sh"]

# Define the codec and create VideoWriter object
#fourcc = cv2.cv.CV_FOURCC(*'DIVX')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret==True:

        # out.write(frame)
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            #Start recording video

            cap2 = cv2.VideoCapture(0)
            out = cv2.VideoWriter('output.avi', -1, 20.0, (640, 480))

            while (cap2.isOpened()):
                ret, frame = cap2.read()

                if ret == True:
                    out.write(frame)
                    cv2.imshow('frame', frame)

                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        #Stop recording video
                        cap2.release()
                        out.release()
                        break
                else:
                    break
    else:
        break

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()