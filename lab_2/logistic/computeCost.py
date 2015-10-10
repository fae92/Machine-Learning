from numpy import *
from sigmoid import sigmoid


def computeCost(theta, X, y):
    # Computes the cost using theta as the parameter
    # for logistic regression.

    m = X.shape[0]  # number of training examples

    # ====================== YOUR CODE HERE ======================
    # Instructions: Calculate the error J of the decision boundary
    # that is described by theta (see the assignment
    #				for more details).


    #computes cost given predicted and actual values
    #m = X.shape[0] #number of training examples


    theta = reshape(theta, (len(theta), 1))  # y = reshape(y,(len(y),1))
    h = sigmoid(X.dot(theta))

    J = (-1. / m) * (transpose(y).dot(log(h)) + transpose(1 - y).dot(log(1-h)))
    #grad = transpose((1. / m) * transpose(sigmoid(X.dot(theta)) - y).dot(X))
    return J

    # end of the function