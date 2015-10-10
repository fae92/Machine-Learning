from numpy import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc

# Load the "gatlin" image data
X = loadtxt('gatlin.csv', delimiter=',')

# Perform SVD decomposition
## TODO: Perform SVD on the X matrix
U,S,V = linalg.svd(X)

# Plot the original image
plt.figure(1)
plt.imshow(X,cmap = cm.Greys_r)
plt.title('Original image (rank 480)')
plt.axis('off')
plt.draw()

# Matrix reconstruction using the top k = [10, 20, 50, 100, 200] singular values
## TODO: Create four matrices X10, X50, X100, X200 for each low rank approximation
X5 = dot(dot(U[:,:5],diag(S)[:5,:5]),V[:5,:])
X20 = dot(dot(U[:,:20],diag(S)[:20,:20]),V[:20,:])
X50 = dot(dot(U[:,:50],diag(S)[:50,:50]),V[:50,:])
X100 = dot(dot(U[:,:100],diag(S)[:100,:100]),V[:100,:])
X200 = dot(dot(U[:,:200],diag(S)[:200,:200]),V[:200,:])


# Error of approximation
## TODO: Compute and print the error of each low rank approximation of the matrix
print ("Error for k=5 ", (linalg.norm(X-X5) / linalg.norm(X)))
print ("Error for k=20 ", (linalg.norm(X-X20) / linalg.norm(X)))  
print ("Error for k=50 ", (linalg.norm(X-X50) / linalg.norm(X)))
print ("Error for k=100 ", (linalg.norm(X-X100) / linalg.norm(X)))
print ("Error for k=200 ", (linalg.norm(X-X200) / linalg.norm(X)))


# Plot the optimal rank-k approximation for various values of k)
# Create a figure with 6 subfigures
plt.figure(2)

# Rank 10 approximation
plt.subplot(321)
plt.imshow(X5,cmap = cm.Greys_r)
plt.title('Best rank' + str(5) + ' approximation')
plt.axis('off')

# Rank 20 approximation
plt.subplot(322)
plt.imshow(X20,cmap = cm.Greys_r)
plt.title('Best rank' + str(20) + ' approximation')
plt.axis('off')

# Rank 50 approximation
plt.subplot(323)
plt.imshow(X50,cmap = cm.Greys_r)
plt.title('Best rank' + str(50) + ' approximation')
plt.axis('off')

# Rank 100 approximation
plt.subplot(324)
plt.imshow(X100,cmap = cm.Greys_r)
plt.title('Best rank' + str(100) + ' approximation')
plt.axis('off')

# Rank 200 approximation
plt.subplot(325)
plt.imshow(X200,cmap = cm.Greys_r)
plt.title('Best rank' + str(200) + ' approximation')
plt.axis('off')

# Original
plt.subplot(326)
plt.imshow(X,cmap = cm.Greys_r)
plt.title('Original image (Rank 480)')
plt.axis('off')

plt.draw()

# Plot the singular values of the original matrix
## TODO: Plot the singular values of X versus their rank k
plt.figure(3) 
plt.plot(S,linewidth=5)
plt.xlabel('k'), 
plt.ylabel(r'$\sigma_k$')
plt.title('Singular values of the "gatlin" image matrix')
plt.draw()

plt.show() 