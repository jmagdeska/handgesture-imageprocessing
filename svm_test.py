from sklearn.svm import SVC
import cv2
import numpy as np

dir_center = "frames_thresh_center/"

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

w_max=300
h_max=400

data_train1 =[]
data_train2 =[]
data_train3 =[]
data_train4 =[]
data_train5 =[]

train1 = []
train2 = []
train3 = []
train4 = []
train5 = []

data_test1 = []
data_test2 = []
data_test3 = []
data_test4 = []
data_test5 = []
data_test6 = []

###### raspredeli mnozestvo na trening i test

for i in range(31):
    for j in range(1, 14):
        for k in range(1, 21):
            if(k > 0 and k < 5):
                data_train1.append(letters[i] + "-" + str(j) + "-" + str(k) + ".png")
            elif (k > 4 and k < 9):
                data_train2.append(letters[i] + "-" + str(j) + "-" + str(k) + ".png")
            elif (k > 8 and k < 13):
                data_train3.append(letters[i] + "-" + str(j) + "-" + str(k) + ".png")
            elif (k > 12 and k < 17 ):
                data_train4.append(letters[i] + "-" + str(j) + "-" + str(k) + ".png")
            elif (k > 16):
                data_train5.append(letters[i] + "-" + str(j) + "-" + str(k) + ".png")

for d in data_train1:
    train1.append(d)

for d in data_train2:
    train1.append(d)
#
# for d in data_train3:
#     train1.append(d)
#
# for d in data_train4:
#     train1.append(d)

training_set = []
testing_set = []
training_y = [] #-klasite odnosno bukite

# print(training_set)

######### dodeli trening Y

for c in range(2):
    for i in range(31):
        for j in range(52): # od sekoja bukva 4 primeroka od 13 dataseta
            training_y.append(i)

cross_test_y = []
#
# ####### dodeli cross_test_y = []; spored test datata stho sme ja dale.
#
for j in range(31):
    for k in range(52):  # od sekoja bukva 4 primeroka od 13 dataseta
        cross_test_y.append(j)

for file in train1:
    print("File: " + file)
    img = cv2.imread(dir_center + file, 0)
    xarr = np.squeeze(np.array(img).astype(np.float32))
    m, v = cv2.PCACompute(xarr, mean=None)
    arr = np.array(v)
    flat_arr = arr.ravel()
    training_set.append(flat_arr)

for file in data_train5:
    print("File test: " + file)
    img = cv2.imread(dir_center + file, 0)
    xarr = np.squeeze(np.array(img).astype(np.float32))
    m, v = cv2.PCACompute(xarr, mean=None)
    arr = np.array(v)
    flat_arr = arr.ravel()
    testing_set.append(flat_arr)

print len(training_set), len(training_y)

trainData = np.float32(training_set)
responses = np.array(training_y)

print "Done with new_listing"

crossTrain = np.float32(trainData)

model = SVM(C=10, gamma=0.018)
model.train(trainData, np.array(training_y))

#
testData = np.float32(testing_set)
crossTest = np.float32(testData)
y_out = model.predict(crossTest)
#
print y_out
# #
total = 0
number = 1612 #number of test examples

for x in xrange(number):
    if y_out[x] == cross_test_y[x]:
        total += 1

print "Percentage is " + str((total/float(number))*100) + "%"
