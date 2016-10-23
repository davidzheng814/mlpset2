from numpy import *
import pylab as pl
import sys

from plotBoundary import *
from optimizers import *

# parameters
part = sys.argv[1]

if part == 'p1':
    X = array([[2, 2], [2, 3], [0, -1], [-3, -2]])
    Y = array([1, 1, -1, -1])
    alpha = svm(X, Y)
    print "Alpha:", alpha
    print "Values:", [predict_svm(X, Y, alpha, val) for val in X]
elif part == 'p2':
    for i in range(1, 5):
        cache_b = None
        train = loadtxt('data/data' + str(i) + '_train.csv')
        X = train[:,0:2].copy()
        Y = train[:,2:3].T[0].copy()
        validate = loadtxt('data/data' + str(i) + '_validate.csv')
        X_val = validate[:,0:2].copy()
        Y_val = validate[:,2:3].copy()
        alpha = svm(X, Y, C=1)

        Xt, Yt, alphat = zip(*[(x, y, a_i) for x, y, a_i in zip(X, Y, alpha) if a_i > 1e-8])
        print "Training Error:", evaluate_svm(X, Y, Xt, Yt, alphat)
        print "Validation Error:", evaluate_svm(X_val, Y_val, Xt, Yt, alphat)
        fn = lambda x: predict_svm(Xt, Yt, alphat, x)
        plotDecisionBoundary(X, Y, fn, [0, -1, 1], title="Training Set " + str(i))
        plotDecisionBoundary(X_val, Y_val, fn, [0, -1, 1], title="Validation Set " + str(i))
elif part == 'p3':
    for i in range(4, 5):
        print "Dataset:", i
        cache_b = None
        train = loadtxt('data/data' + str(i) + '_train.csv')
        X = train[:,0:2].copy()
        Y = train[:,2:3].T[0].copy()
        validate = loadtxt('data/data' + str(i) + '_validate.csv')
        X_val = validate[:,0:2].copy()
        Y_val = validate[:,2:3].copy()
        optimal_val_error = 10
        optimal_C = 0
        optimal_K = 0
        for C in [0.01, 0.1, 1, 10, 100]:
            print "C:", C
            for j, k in enumerate([dot_product, rbf_factory(0.5), rbf_factory(1), rbf_factory(2)]):
                print "K:", j
                alpha = svm(X, Y, C=C, k=k)
                Xt, Yt, alphat = zip(*[(x, y, a_i) for x, y, a_i in zip(X, Y, alpha) if a_i > 1e-8])
                print "Training Error:", evaluate_svm(X, Y, Xt, Yt, alphat, k=k)
                val_error = evaluate_svm(X_val, Y_val, Xt, Yt, alphat, k=k)
                print "Validation Error:", val_error
                fn = lambda x: predict_svm(Xt, Yt, alphat, x, k=k)
                if optimal_val_error > val_error:
                    optimal_val_error = val_error
                    optimal_C = C
                    optimal_K = j
        print "Optimal: ", i, optimal_val_error, optimal_C, optimal_K
