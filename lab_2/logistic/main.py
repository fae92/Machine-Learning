from numpy import *
import matplotlib.pyplot as plt
import scipy.optimize as op
from computeCost import computeCost
from computeGrad import computeGrad
from predict import predict
from pylab import scatter, show, legend, xlabel, ylabel


def map_feature(x):
    x.inser(0, 1)
    return x


nbColumns = 10


# Load the dataset
data = loadtxt('breast_cancer.txt', delimiter=',')

# ====================== YOUR CODE HERE ======================
# Instructions: Implement the main program following 
#the instructions given to you.
#  
# =============================================================

random.shuffle(data)  # shuffle datataset

#revome the id or begin in the second column
X = data[:, 1:nbColumns+1]
y = data[:, nbColumns]
#print y

#Add intercept term to X
#X_new = ones ( ( X.shape[0] , c ) )
#X_new[ : , 1:c ] = X
#X = X_new

#partitionning of data
NB_partitions = 10
m, c = X.shape
lenOfPartition = int(m / (NB_partitions * 1.0))

#make a list of thresholders
thres = arange(0, 1.0, 0.1)

#initialize the False Positive and the True Positive vectors
FalseP = zeros(( len(thres) , 1 ))
TrueP = zeros(( len(thres) , 1 ))

#Loop through the partitions
for i in range(NB_partitions):

    # Get the partition used for evaluation
    X_eval = ones((lenOfPartition,nbColumns))
    X_eval[:,1:nbColumns] = X[lenOfPartition*i:lenOfPartition*(i+1), 0:nbColumns-1]
    y_eval = X[lenOfPartition*i:lenOfPartition*(i+1), nbColumns-1]

    # Get the training data (9 partitions)
    X_train = ones((m-lenOfPartition, nbColumns))
    y_train = zeros((m-lenOfPartition, 1))
    a = X[0:lenOfPartition*i,0:nbColumns-1]
    l = a.shape[0]
    X_train[0:l, 1:nbColumns] = a[:,:]
    y_train[0:l, 0] = X[0:lenOfPartition*i, nbColumns-1]
    X_train[l:, 1:nbColumns] = X[lenOfPartition*(i+1):, 0:nbColumns-1]
    y_train[l:,0] = X[lenOfPartition*(i+1):, nbColumns-1]
    #print X_train
    #print y_train

    # Start the training of the 9 partitions
    # Initialize fitting parameters
    initial_theta = zeros((nbColumns,1))
    # Run minimize ( ) to obtain the optimal theta
    Result = op.minimize(fun = computeCost, x0 = initial_theta, args = (X_train,y_train), method = 'tnc', jac = computeGrad)
    theta = Result.x
    #print theta



    i = 0
    # Loop through the m steps between 0 and 1 for all the thresholds
    for ithres in thres:
        p = predict(theta,X_eval,ithres)
        #print y_eval

        # take the position( index in array ) of people who is positive in the position
        pos = where(p == 1)[0]
        # Compute the false positive rates

        #loop trought the ids which are positive
        for id in pos:
            if y_eval[id]==0: #false positive
                FalseP[i] += 1
            else:
                TrueP[i] += 1

        i = i +1


# Compute positives and negatives set
Positive =  len(where(y == 1)[0])
Negative = len(where(y == 0)[0])


#print FalseP * 1. / Negative , TrueP  / Positive

plt.plot(FalseP * 1./ Negative , TrueP *1. / Positive , marker='D', color='r')
plt.plot ([0.8 , 1], [0.8 , 1] )
plt.xlabel('False Positive Rate (1-Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title("ROC Curve", fontsize=14 )

show()