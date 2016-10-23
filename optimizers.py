from sklearn.linear_model import LogisticRegression
from cvxopt import matrix, solvers
import numpy as np
from numpy import linalg as LA
import math

def logistic_regression(X, Y, penalty='l2', lam=0, verbose=0):
    C = 1 / float(lam) if lam != 0 else 10000
    reg = LogisticRegression(penalty=penalty, C=C, verbose=verbose,
                             intercept_scaling=1)
    reg.fit(X, Y)
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
