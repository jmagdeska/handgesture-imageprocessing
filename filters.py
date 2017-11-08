import cv2
import imutils
import numpy as np
from scipy import ndimage

w_max=400
h_max=250


letters = ["a", "b", "c", "ch", "d", "dj", "dz", "e", "f", "g", "gj", "h", "i", "j", "k", "kj", "l", "lj", "m", "n",
           "nj", "o", "p", "r", "s", "sh", "t", "u", "v", "z", "zh"]

for i in range(31):
    for j in range(1,10):
        single = cv2.imread("C:\Users\Kosara\Documents\DIPLOMA THESIS\handgesture-imageprocessing-master/frames skin/" +letters[i] + "-1-" + format(j) + '.png',0)
        result = cv2.Sobel(single, cv2.CV_64F, 1, 0, ksize=5)
        result1 = cv2.Sobel(result, cv2.CV_64F, 0, 1, ksize=5)
        abs_result = np.absolute(result1)
        final_result = np.uint8(abs_result)
        blurred1 = cv2.GaussianBlur(final_result, (5, 5), 0)
        # cv2.imwrite(
        #     "C:\Users\Kosara\Documents\DIPLOMA THESIS\handgesture-imageprocessing-master/frames sobel/" + letters[
        #         i] + "-1-" + format(j) + ".png", final_result)

        _, thresh1 = cv2.threshold(blurred1, 127, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cv2.imwrite("C:\Users\Kosara\Documents\DIPLOMA THESIS\handgesture-imageprocessing-master/frames thresh as/" + letters[i] + "-1-" + format(j) + ".png", thresh1)

        cnts = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        # loop over the contours
        for c in cnts:
            # compute the center of the contour
            M = cv2.moments(c)

            if M["m00"] > 5000:
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

                print format(h_max/2) + " " + format(h/2) + " " + format(w_max/2) + " " + format(w/2) + "   " +format(letters[i])

                black[h_max/2 - h/2 - offset_h: h_max/2 + h/2, w_max/2 - w/2 - offset_w: w_max/2 + w/2] = thresh1[y:y+h, x:x+w]
                # cv2.imshow("Cropped gesture",thresh1[y:y+h, x:x+w])
                # cv2.waitKey(0)
                cv2.imwrite("C:\Users\Kosara\Documents\DIPLOMA THESIS\handgesture-imageprocessing-master/frames thresh center as/" + letters[i] + "-1-" + format(j) + ".png", black)

            else:
                cX, cY = 0, 0

print "Done with processing the images"