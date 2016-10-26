from numpy import *
from plotBoundary import *
import pylab as pl
from optimizers import *
import csv
from pylab import imshow, show, cm
 
imshow(digit.reshape(28,28),cmap=cm.gray)
show()

# train[digit] gives the training set of points for the given digit
normalized_train = []
normalized_val = []
normalized_test = []
unnormalized_train, unnormalized_val, unnormalized_test = [], [], []
for i in range(10):
    with open('data/mnist_digit_' + str(i) + '.csv', 'r') as fin:
        reader=csv.reader(fin)
        data = []
        unnormalized_data = []
        rowNum = 0
        for row in reader:
            data.append([double(s) * 2/255 - 1 for s in row[0].split()])
            unnormalized_data.append([double(s) for s in row[0].split()])
            rowNum += 1
            if (rowNum >= 500):
                break
    normalized_train.append(data[:200])
    normalized_val.append(data[200:350])
    normalized_test.append(data[350:500])
    unnormalized_train.append(unnormalized_data[:200])
    unnormalized_val.append(unnormalized_data[200:350])
    unnormalized_test.append(unnormalized_data[350:500])
print "Data all loaded\n"

data_choices = [(normalized_train, normalized_val, normalized_test),
                (unnormalized_train, unnormalized_val, unnormalized_test)]

digit_choices = [([1], [7]),
                 ([3], [5]),
                 ([4], [9]),
                 ([0,2,4,6,8], [1,3,5,7,9])]

for (train, val, test) in data_choices:
    print "\n--------------- TRYING OUT DIFFERENT NORMALIZATION--------------\n"
    for (digits_1, digits_2) in digit_choices:
        print "\n---------------TRYING OUT DIGITS ", digits_1, "AND", digits_2, "---------------\n"

        pos_train, neg_train, pos_val, neg_val, pos_test, neg_test = [], [], [], [], [], []
        for digit in digits_1:
            pos_train += train[digit]
            pos_val += val[digit]
            pos_test += test[digit]
        for digit in digits_2:
            neg_train += train[digit]
            neg_val += val[digit]
            neg_test += test[digit]

        print "\nTRAINING LOGISTIC REGRESSION for digits ", digits_1, "AND", digits_2, "\n"
        train_logistic_regression(pos_train, neg_train, pos_val, neg_val, pos_test, neg_test)
        print "\nTRAINING LINEAR SVM for digits ", digits_1, "AND", digits_2, "\n"
        train_linear_svm(pos_train, neg_train, pos_val, neg_val, pos_test, neg_test)
        print "\nTRAINING GAUSSIAN SVM for digits ", digits_1, "AND", digits_2, "\n"
        train_gaussian_svm(pos_train, neg_train, pos_val, neg_val, pos_test, neg_test)
        print "\nTRAINING GAUSSIAN PEGASOS for digits ", digits_1, "AND", digits_2, "\n"
        train_gaussian_pegasos(pos_train, neg_train, pos_val, neg_val, pos_test, neg_test)