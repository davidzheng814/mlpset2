from numpy import *
from plotBoundary import *
import pylab as plt
from optimizers import *
import math

# load data from csv files
train = loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

font = {'size':19}
plt.rc('font', **font)

# Creates a graph that compares lambda to margin size
# lam_data = range(-10, 2, 1)
# margin_data = []
# for x in lam_data:
# 	w = pegasos_no_kernel(X, Y, lam=math.pow(2, x))
# 	margin_data.append(1 / linalg.norm(w[1:]))
# plt.plot(lam_data, margin_data,'--o')
# plt.xlim([-11,2])
# plt.xlabel("$n$")
# plt.ylabel("Margin size")
# plt.title("Margin size of Pegasos with $\lambda=2^n$")
# plt.show()

# Plots the decision boundary for given lambda
i = -4
lam = math.pow(2, i)
w = pegasos_no_kernel(X, Y, lam=lam)
def predict_linearSVM(x):
	return w[0] + w[1:].dot(x)
plotDecisionBoundary(X, Y, predict_linearSVM, [-1,0,1], title = 'Linear SVM using $\lambda$=2^(' + str(i) + ')')
