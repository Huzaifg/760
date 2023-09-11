import numpy as np
import matplotlib.pyplot as plt

# Question 7b
#Make a scatter plot by drawing 100 items from a mixture distribution  $0.3 N\left((5, 0)^{T}, \begin{pmatrix} 1 & 0.25 \\ 0.25 & 1\\ \end{pmatrix}\right) +0.7 N\left((-5, 0)^{T}, \begin{pmatrix} 1 & -0.25 \\ -0.25 & 1\\ \end{pmatrix}\right)$.

# Set good fonts for plot
plt.rc('font', family='serif')
# Set font size for title, labes and ticks

plt.rcParams.update({'font.size': 12})   

# Draw from the comined distribution
#Combined covariance matrix
cov = 0.3**2 * np.array([[1,0.25],[0.25,1]]) + 0.7**2 * np.array([[1,-0.25],[-0.25,1]])
# Combined mean
mean = 0.3 * np.array([5,0]) + 0.7 * np.array([-5,0])
X = np.random.multivariate_normal(mean=mean, cov = cov, size = 100)

# Plot the scatter plot
plt.scatter(X[:,0], X[:,1])

# Add X_1 and X_2 labels in the plot with latex font
plt.xlabel(r'$X_1$')
plt.ylabel(r'$X_2$')

# ADD a title
plt.title("Scatter plot of 100 items from a mixture distribution")

#asve the plot
plt.savefig("./images/7_b.png")