import cv2
import numpy as np
from skimage.feature import hog
import random

dir_center = "frames_skin/"

letters = ["a", "b", "v", "g", "d", "gj", "e", "zh", "z", "dj", "i", "j", "k", "l", "lj", "m", "n", "nj", "o", "p", "r", "s", "t", "kj", "u", "f", "h", "c", "ch", "dz", "sh"]

class StatModel(object):
    def load(self, fn):
        self.model.load(fn)  # Known bug: https://github.com/opencv/opencv/issues/4969
    def save(self, fn):
        self.model.save(fn)

class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()

def randomize(a, b):
    # Generate the permutation index array.
    permutation = np.random.permutation(a.shape[0])
    # Shuffle the arrays by giving the permutation in the square brackets.
    shuffled_a = a[permutation]
    shuffled_b = b[permutation]
    return shuffled_a, shuffled_b

data1 =[]
data2 =[]
data3 =[]
data4 =[]
data5 =[]

data_train = []
data_test = []

# separate data into 5 parts, 4 will be used for training, 1 for testing
for i in range(31):
    for j in range(1, 14):
        for k in range(1, 21):
            if(k > 0 and k < 5):
                data1.append(letters[i] + "-" + str(j) + "-" + str(k) + ".png")
            elif (k > 4 and k < 9):
                data2.append(letters[i] + "-" + str(j) + "-" + str(k) + ".png")
            elif (k > 8 and k < 13):
                data3.append(letters[i] + "-" + str(j) + "-" + str(k) + ".png")
            elif (k > 12 and k < 17 ):
                data4.append(letters[i] + "-" + str(j) + "-" + str(k) + ".png")
            elif (k > 16):
                data5.append(letters[i] + "-" + str(j) + "-" + str(k) + ".png")

for d in data1:
    data_train.append(d)
for d in data2:
    data_train.append(d)
for d in data5:
    data_train.append(d)
for d in data4:
    data_train.append(d)

training_set = []
testing_set = []
training_labels = []

for c in range(4):
    for i in range(31):
        for j in range(52): # each letter 4 examples, 13 different datasets
            training_labels.append(i)

# shuffle both arrays (data_train and training_labels) in the same order
list1 = list(zip(data_train, training_labels))
random.shuffle(list1)
data_train, training_labels = zip(*list1)

print "Done with random shuffle of training dataset"

print data_train[:10], training_labels[:10]

cross_test_labels = []

for j in range(31):
    for k in range(52):  # each letter 4 examples, 13 different datasets
        cross_test_labels.append(j)

list2 = list(zip(data3, cross_test_labels))
random.shuffle(list2)
data3, cross_test_labels = zip(*list2)

print "Done with random shuffle of testing dataset"

for file in data_train:
    print("File: " + file)
    img = cv2.imread(dir_center + file, 0)
    hist = hog(img, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    training_set.append(hist)

for file in data3:
    print("File test: " + file)
    img = cv2.imread(dir_center + file, 0)
    hist = hog(img, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)
    testing_set.append(hist)

print "Done with fitting data"

trainData = np.float32(training_set)
testData = np.float32(testing_set)
crossTest = np.float32(testData)

model = SVM(C=50, gamma=0.006)
model.train(trainData, np.array(training_labels))

print "Done with SVM training"

y_out = model.predict(crossTest)

print "Done with SVM prediction"

print y_out

total = 0
number = 1612 #number of test examples

for x in xrange(number):
    if y_out[x] == cross_test_labels[x]:
        total += 1

print "Percentage is " + str((total/float(number))*100) + "%"
