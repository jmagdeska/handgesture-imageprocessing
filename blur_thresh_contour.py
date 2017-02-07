import cv2
import imutils
import numpy as np

w_max=400
h_max=225
letters = ["a", "b", "f", "i", "l", "m", "o", "t", "u", "z"]

for i in range(10):
    for j in range(1,10):
        single = cv2.imread('C:\Users\Jana\PyCharmProjects\skin-detection/frames_skin/' +letters[i] + format(j) + '.png')
        grey1 = cv2.cvtColor(single, cv2.COLOR_BGR2GRAY)
        blurred1 = cv2.GaussianBlur(grey1, (5, 5), 0)
        _, thresh1 = cv2.threshold(blurred1, 127, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imwrite("frames_thresh/" + letters[i] + format(j) + ".png", thresh1)

        cnts = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # loop over the contours
        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)

            if M["m00"] > 2000:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])

                cv2.drawContours(thresh1, [c], -1, (0, 255, 0), 2)

                rect = cv2.boundingRect(c)
                x, y, w, h = rect

                offset_h = 0
                offset_w = 0
                black = np.zeros((h_max,w_max), np.uint8)

                if(h%2 != 0): offset_h = 1
                if(w%2 != 0): offset_w = 1

                black[h_max/2 - h/2 - offset_h: h_max/2 + h/2, w_max/2 - w/2 - offset_w: w_max/2 + w/2] = thresh1[y:y+h, x:x+w]
                # cv2.imshow("Cropped gesture",thresh1[y:y+h, x:x+w])
                # cv2.waitKey(0)
                cv2.imwrite("frames_thresh_center/" + letters[i] + format(j) + ".png", black)

            else:
                cX, cY = 0, 0

print "Done with processing the images"