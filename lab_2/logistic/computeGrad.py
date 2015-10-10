from numpy import *
from sigmoid import sigmoid


def computeGrad(theta, X, y):
    # Computes the gradient of the cost with respect to
    # the parameters.

    m,c = X.shape  # number of training examples
    theta = reshape(theta,(len(theta),1))
    grad = zeros(size(theta))  # initialize gradient

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the gradient of cost for each theta,
    # as described in the assignment.
    h = sigmoid(X.dot(theta))
    delta = h-y
    for j in range(c):
        sumdelta = delta.T.dot(X[:,j])
        grad[j] = (1.0 / m) * sumdelta

    return grad





# =============================================================