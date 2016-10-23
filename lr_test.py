from numpy import *
import pylab as pl
import sys

from plotBoundary import *
from optimizers import *

# import your LR training code

# parameters
part = sys.argv[1]

if part == 'p1':
    train = loadtxt('data/data1_train.csv')
    X = train[:,0:2]
    Y = train[:,2:3].T[0]

    print "LogReg Gradient Descent. Lambda = 0"
    logistic_regression(X, Y, lam=0, verbose=1)
    print "Lambda = 1"
    logistic_regression(X, Y, lam=1, verbose=1)
elif part == 'p2':
    for i in range(1, 5):
        print '======Dataset: ', i, '======'
        train = loadtxt('data/data' + str(i) + '_train.csv')
        X = train[:,0:2]
        Y = train[:,2:3].T[0]
        for penalty in ['l1', 'l2']:
            for lam in arange(0, 5, 0.5):
                reg = logistic_regression(X, Y, penalty=penalty,
                                          lam=lam)
                error = reg.score(X, Y)
                print "Penalty:", penalty, "Lambda:", lam
                print "Coefficient:", reg.coef_, "Intercept:", reg.intercept_
                print "Error:", error, "\n"
elif part == 'p3':
    for i in range(1, 5):
        print '======Dataset: ', i, '======'
        train = loadtxt('data/data' + str(i) + '_train.csv')
        X = train[:,0:2]
        Y = train[:,2:3].T[0]
        validate = loadtxt('data/data' + str(i) + '_validate.csv')
        X_val = validate[:,0:2]
        Y_val = validate[:,2:3]
        test = loadtxt('data/data' + str(i) + '_test.csv')
        X_test = test[:,0:2]
        Y_test = test[:,2:3]
        for penalty in ['l1', 'l2']:
            for lam in arange(0, 5, 0.5):
                reg = logistic_regression(X, Y, penalty=penalty,
                                          lam=lam)
                train_error = reg.score(X, Y)
                val_error = reg.score(X_val, Y_val)
                test_error = reg.score(X_test, Y_test)
                print "Penalty:", penalty, "Lambda:", lam
                print "Coefficient:", reg.coef_, "Intercept:", reg.intercept_
                print "Training Error:", train_error
                print "Validation Error:", val_error
                print "Test Error:", test_error, "\n"
