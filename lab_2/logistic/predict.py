from numpy import *
from sigmoid import sigmoid

def predict(theta, X, thres):
	# Predict whether the label is 0 or 1 using learned logistic 
	# regression parameters theta. The threshold is set at 0.5
	
	#m = X.shape[0] # number of training examples
	

	
	#p = zeros(m) # logistic regression outputs of training examples
	
	
	# ====================== YOUR CODE HERE ======================
    # Instructions: Predict the label of each instance of the
    #				training set.
    

    m, n = X.shape
    #c = zeros(m) # predicted classes of training examples
    p = zeros(shape=(m,1))

    h = sigmoid(X.dot(theta.T))

    for it in range(0, h.shape[0]):
        if h[it] > thres:
            p[it, 0] = 1
        else:
            p[it, 0] = 0

    return p



    
    # =============================================================