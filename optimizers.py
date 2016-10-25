from sklearn.linear_model import LogisticRegression
from cvxopt import matrix, solvers
import numpy as np
from numpy import linalg as LA
import math
solvers.options['show_progress'] = False

def logistic_regression(X, Y, penalty='l2', lam=0, verbose=0):
    C = 1 / float(lam) if lam != 0 else 10000
    reg = LogisticRegression(penalty=penalty, C=C, verbose=verbose,
                             intercept_scaling=1./C)
    reg.fit(X, Y)
    # print reg
    return reg

def dot_product(x, y):
    return x.dot(y)

def rbf_factory(sigma=1):
    return lambda x, y: math.exp(-LA.norm(x - y) ** 2 / (2 * sigma ** 2))

cache_b = None

def get_svm_b(X, Y, alpha, k=dot_product):
    global cache_b
    if cache_b is not None:
        return cache_b
    count = 0
    b = 0
    eps = 1e-8
    for x, y, alpha_i in zip(X, Y, alpha):
        if alpha_i < eps:
            continue
        count += 1
        for x2, y2, alpha_j in zip(X, Y, alpha):
            b -= alpha_j * y2 * k(x, x2)
        b += y
    b /= count
    cache_b = b
    return b

def predict_svm(X, Y, alpha, val, k=dot_product):
    b = get_svm_b(X, Y, alpha, k=k)
    sum = 0
    for x, y, alpha_i in zip(X, Y, alpha):
        sum += alpha_i * y * k(x, val)
    return sum + b

def evaluate_svm(X_test, Y_test, X, Y, alpha, k=dot_product):
    correct = 0
    for x, y in zip(X_test, Y_test):
        predict_y = predict_svm(X, Y, alpha, x, k=k)
        predict_y = 1 if predict_y > 0 else -1
        if predict_y == y:
            correct += 1
    return 1 - correct / float(len(X_test))

def svm(X, Y, k=dot_product, C=10000):
    N = len(Y)
    P = np.zeros((N, N))
    q = np.zeros(N)
    G = np.zeros((2 * N, N))
    h = np.zeros(2 * N)
    A = np.zeros((1, N))
    b = np.zeros(1)

    for i in range(N):
        G[i, i] = -1
        G[N + i, i] = 1
        h[N + i] = C
        A[0, i] = Y[i]
        q[i] = -1

    for i, (x1, y1) in enumerate(zip(X, Y)):
        for j, (x2, y2) in enumerate(zip(X, Y)):
            P[i, j] = y1 * y2 * k(x1, x2)

    P, q, G, h, A, b = [matrix(x, tc='d') for x in [P, q, G, h, A, b]]
    sol = solvers.qp(P, q, G, h, A, b)
    print "Objective:", sol['primal objective']
    return sol['x']

def pegasos_no_kernel(X, Y, lam=1, max_epochs=1000):
    N = len(Y)
    X = np.insert(X, 0, np.ones(N), axis=1)
    Y = np.ndarray.flatten(Y)
    t = 0
    w = np.zeros(len(X[0]))
    for epoch in range(max_epochs):
        for i in range(N):
            t += 1
            step_size = 1. / (t * lam)
            old_bias = w[0]
            new_w = (1 - step_size * lam) * w
            new_w[0] = old_bias
            if (Y[i] * (w.dot(X[i])) < 1):
                w = new_w + step_size * Y[i] * X[i]
            else:
                w = new_w
    return w

def pegasos_with_kernel(X, Y, K, lam=1, max_epochs=1000):
    N = len(Y)
    t = 0
    D = len(X[0])
    Y = np.ndarray.flatten(Y)
    alpha = np.zeros(N)
    for epoch in range(max_epochs):
        for i in range(N):
            t += 1
            step_size = 1. / (t * lam)
            if (Y[i] * alpha.dot(K[i]) < 1 ):
                alpha[i] = (1 - step_size * lam) * alpha[i] + step_size * Y[i]
            else:
                alpha[i] = (1 - step_size * lam) * alpha[i]
    return alpha

def get_np_array_format(pos, neg):
    num = len(pos)
    X = np.asarray(pos + neg)
    Y = np.append(np.ones(num), np.zeros(num) - np.ones(num))
    return (X, Y)

def train_logistic_regression(pos_train, neg_train, pos_val, neg_val, pos_test, neg_test):
    X_train, Y_train = get_np_array_format(pos_train, neg_train)
    X_val, Y_val = get_np_array_format(pos_val, neg_val)
    X_test, Y_test = get_np_array_format(pos_test, neg_test)
    best_val_error = float("inf")
    best_penalty = None
    best_lam = None
    best_reg = None
    for penalty in ['l1', 'l2']:
        for lam in [1e-2, 1e-1, 1e0, 1e1, 1e2]:
            reg = logistic_regression(X_train, Y_train, penalty=penalty, lam=lam)
            val_error = 1 - reg.score(X_val, Y_val)
            # print "Coefficient:", reg.coef_, "Intercept:", reg.intercept_
            print "Penalty", penalty
            print "Lambda", lam
            print "Validation Error:", val_error, "\n"
            if (val_error < best_val_error):
                best_val_error = val_error
                best_penalty = penalty
                best_lam = lam
                best_reg = reg
    print "Optimal penalty:", best_penalty
    print "Optimal lambda:", best_lam
    print "Training Error:", 1 - best_reg.score(X_train, Y_train)
    print "Validation Error:", 1 - best_reg.score(X_val, Y_val)
    print "Test Error:", 1 - best_reg.score(X_test, Y_test)

def train_linear_svm(pos_train, neg_train, pos_val, neg_val, pos_test, neg_test):
    X_train, Y_train = get_np_array_format(pos_train, neg_train)
    X_val, Y_val = get_np_array_format(pos_val, neg_val)
    X_test, Y_test = get_np_array_format(pos_test, neg_test)
    best_val_error = float("inf")
    best_alpha = None
    best_C = None
    for C in [1e-2, 1e-1, 1e0, 1e1, 1e2]:
        alpha = svm(X_train, Y_train, C=C)
        val_error = evaluate_svm(X_val, Y_val, X_train, Y_train, alpha)
        print "C:", C
        print "Validation Error:", val_error, "\n"
        if (val_error < best_val_error):
            best_val_error = val_error
            best_alpha = alpha
            best_C = C
    print "Optimal C:", best_C
    print "Training Error", evaluate_svm(X_train, Y_train, X_train, Y_train, best_alpha)
    print "Validation Error", evaluate_svm(X_val, Y_val, X_train, Y_train, best_alpha)
    print "Test Error:", evaluate_svm(X_test, Y_test, X_train, Y_train, best_alpha)

def train_gaussian_svm(pos_train, neg_train, pos_val, neg_val, pos_test, neg_test):
    X_train, Y_train = get_np_array_format(pos_train, neg_train)
    X_val, Y_val = get_np_array_format(pos_val, neg_val)
    X_test, Y_test = get_np_array_format(pos_test, neg_test)
    best_val_error = float("inf")
    best_alpha = None
    best_C = None
    best_gamma = None
    for C in [1e-2, 1e-1, 1e0, 1e1, 1e2]:
        for gamma in [1e-2, 1e-1, 1e0, 1e1, 1e2]:
            k = rbf_factory(sigma=gamma)
            alpha = svm(X_train, Y_train, C=C, k=k)
            val_error = evaluate_svm(X_val, Y_val, X_train, Y_train, alpha, k=k)
            print "C:", C
            print "Gamma:", gamma
            print "Validation Error:", val_error, "\n"
            if (val_error < best_val_error):
                best_val_error = val_error
                best_alpha = alpha
                best_C = C
                best_gamma = gamma
    print "Optimal C:", best_C
    print "Optimal Gamma:", best_gamma
    k = rbf_factory(sigma=best_gamma)
    print "Training Error:", evaluate_svm(X_train, Y_train, X_train, Y_train, best_alpha, k=k)
    print "Validation Error:", evaluate_svm(X_val, Y_val, X_train, Y_train, best_alpha, k=k)
    print "Test Error:", evaluate_svm(X_test, Y_test, X_train, Y_train, best_alpha, k=k)

def make_gram_matrix(N, k, X):
    K = np.empty([N, N])
    for i in range(N):
        for j in range(i, N):
            K[i][j] = k(X[i], X[j])
            K[j][i] = K[i][j]
    return K

def evaluate_pegasos(X_test, Y_test, X_train, Y_train, alpha, k=dot_product):
    N = len(alpha)
    def predict_gaussianSVM(x):
        return sum([alpha[i] * k(X_train[i], x) for i in range(N)])
    correct = 0
    for x, y in zip(X_test, Y_test):
        predict_y = predict_gaussianSVM(x)
        predict_y = 1 if predict_y > 0 else -1
        if predict_y == y:
            correct += 1
    return 1 - correct / float(len(X_test))

def train_gaussian_pegasos(pos_train, neg_train, pos_val, neg_val, pos_test, neg_test):
    X_train, Y_train = get_np_array_format(pos_train, neg_train)
    X_val, Y_val = get_np_array_format(pos_val, neg_val)
    X_test, Y_test = get_np_array_format(pos_test, neg_test)
    best_val_error = float("inf")
    best_alpha = None
    best_gamma = None
    best_lam = None
    for lam in [1e-3, 1e-2, 1e-1, 1e0, 1e1]:
        for gamma in [1e-2, 1e-1, 1e0, 1e1, 1e2]:
            k = rbf_factory(sigma=gamma)
            alpha = pegasos_with_kernel(X_train, Y_train, make_gram_matrix(len(X_train), k, X_train), lam=lam)
            val_error = evaluate_pegasos(X_val, Y_val, X_train, Y_train, alpha, k=k)
            print "Lambda:", lam
            print "Gamma:", gamma
            print "Validation Error:", val_error, "\n"
            if (val_error < best_val_error):
                best_val_error = val_error
                best_alpha = alpha
                best_lam = lam
                best_gamma = gamma
    print "Optimal Lambda:", best_lam
    print "Optimal Gamma:", best_gamma
    k = rbf_factory(sigma=best_gamma)
    print "Training Error:", evaluate_pegasos(X_train, Y_train, X_train, Y_train, best_alpha, k=k)
    print "Validation Error:", evaluate_pegasos(X_val, Y_val, X_train, Y_train, best_alpha, k=k)
    print "Test Error:", evaluate_pegasos(X_test, Y_test, X_train, Y_train, best_alpha, k=k)