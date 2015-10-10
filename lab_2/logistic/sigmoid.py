import numpy as np
from math import e
from math import pow

def sigmoid(z):
	# Computes the sigmoid of z.

	# ====================== YOUR CODE HERE ======================
    # Instructions: Implement the sigmoid function as given in the
    # assignment.
    g = 1.0 / ( 1.0 + np.exp(-z))
    
    # =============================================================
             
    return g
