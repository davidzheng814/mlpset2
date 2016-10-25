from numpy import *
from plotBoundary import *
import pylab as plt
from optimizers import *

# load data from csv files
train = loadtxt('data/data3_train.csv')
X = train[:,0:2]
Y = train[:,2:3]

# Carry out training.
epochs = 1000;
lmbda = .02;
gamma = 1;

# Create the K matrix
N = len(Y)
# kernel = rbf_factory(sigma=gamma)
# K = np.empty([N, N])
# for i in range(N):
# 	for j in range(i, N):
# 		K[i][j] = kernel(X[i], X[j])
# 		K[j][i] = K[i][j]

# # Create the predict_gaussianSVM model
# alpha = pegasos_with_kernel(X, Y, K, lam=lmbda, max_epochs=epochs)
# num_support_vectors = sum([1 if alpha[i]!=0 else 0 for i in range(N)])
# print num_support_vectors
# def predict_gaussianSVM(x):
#     return sum([alpha[i] * kernel(X[i], x) for i in range(N)])

# # plot training results
# plotDecisionBoundary(X, Y, predict_gaussianSVM, [-1,0,1], title = 'Gaussian Kernel SVM using $\gamma$=' + str(gamma))

# alpha_2 = svm(X, Y, C=1, k=kernel)
# Xt, Yt, alphat = zip(*[(x, y, a_i) for x, y, a_i in zip(X, Y, alpha_2) if a_i > 1e-8])
# plotDecisionBoundary(X, Y, lambda x: predict_svm(Xt, Yt, alphat, x, k=kernel), [-1,0,1])
# plt.show()

# Create the graph of num support vectors to gamma
# gamma = range(-2, 3, 1)
gamma = [-4, -3.5, -3, -2.5, -2, -1.5, -1, -.5, 0, .5, 1, 1.5, 2, 2.5, 3., 3.5, 4]
num_support_vectors = []
for x in gamma:
	# make kernel and K
	kernel = rbf_factory(sigma=math.pow(2, x))
	K = np.empty([N, N])
	for i in range(N):
		for j in range(i, N):
			K[i][j] = kernel(X[i], X[j])
			K[j][i] = K[i][j]
	# solve
	alpha = pegasos_with_kernel(X, Y, K, lam=lmbda, max_epochs=epochs)
	# alpha = svm(X, Y, C=100, k=kernel)
	print alpha
	num_support_vectors.append(sum([1 if alpha[i]!=0 else 0 for i in range(N)]))
# num_support_vectors = [100, 38, 30, 30, 40]
plt.plot(gamma, num_support_vectors,'--o')
# plt.ylim([20,110])
plt.xlim([-4.5,4.5])
plt.xlabel("$n$")
plt.ylabel("Number of support vectors")
plt.title("Number of support vectors with $\gamma=2^n$")
plt.show()
