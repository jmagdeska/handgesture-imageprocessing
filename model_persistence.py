import cv2
from skimage.feature import hog
import pickle

dir = "frames_skin/a-8-10.png"
img = cv2.imread(dir, 0)
hist = hog(img, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)

testing_set = []
testing_set.append(hist)

clf = pickle.load(open('pickled_data', 'rb'))
y = clf.predict(testing_set)

print y