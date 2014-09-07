from numpy import loadtxt, where, zeros, e, array, log, ones, mean, where
from pylab import scatter, show, legend, xlabel, ylabel, plot
from scipy.optimize import fmin_bfgs
import pandas as pd

def computeCost(theta, X, y):
    theta.shape = (1, 3)
    m = y.size
    z=X.dot(theta.T)
    h = 1.0 / (1.0 + e ** (-1.0 * z))
    J = (1.0 / m) * ((-y.T.dot(log(h))) - ((1.0 - y.T).dot(log(1.0 - h))))
    return 1 * J.sum()

def gradientDescent(theta,alpha, X, y):
    theta.shape = (1, 3)
    
    grad = zeros(3)
    z=X.dot(theta.T)
    h = 1.0 / (1.0 + e **(-1.0 * z))

    delta = h - y
    l = grad.size
    theta.shape = (3,)
    for i in range(l):
        sumdelta = delta.T.dot(X[:, i])
        grad[i] = theta[i] - alpha * sumdelta
    
    return  grad

def plotInputData(data):
    X = data[:, 0:2]
    y = data[:, 2]
    pos = where(y == 1)
    neg = where(y == 0)
    scatter(X[pos, 0], X[pos, 1], marker='o', c='b')
    scatter(X[neg, 0], X[neg, 1], marker='x', c='r')
    xlabel('Exam 1 score')
    ylabel('Exam 2 score')
    legend(['Not Admitted', 'Admitted'])
    #show()
        
def main():
    #load the dataset
    data = loadtxt('ex2data1.txt', delimiter=',')

    #plot input data
    plotInputData(data)
    #features manipulation
    X = data[:, 0:2]
    y = data[:, 2]
    m, n = X.shape
    y.shape = (m, 1)
    #Add intercept term to x and X_test
    it = ones(shape=(m, 3))
    it[:, 1:3] = X

    #Initialize theta parameters
    theta=zeros(3)

    print "initial cost: "+ str(computeCost(theta,it,y))

    #Some gradient descent settings
    iterations = 15000
    alpha = 0.0001
    for i in range(iterations):
	    theta= gradientDescent(theta,alpha, it, y)

    print "Theta value is: "+ str(theta)

    print "final cost: "+ str(computeCost(theta,it,y))


if __name__ == '__main__':
  main()
