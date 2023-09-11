import numpy as np
import matplotlib.pyplot as plt
 
# Make a scatter plot by drawing 100 items from a two dimensional Gaussian $N((1, -1)^{T}, 2I)$, where I is an identity matrix in $\mathbb{R}^{2 \times 2}$

# Set good fonts for plot
plt.rc('font', family='serif')
# Set font size for title, labes and ticks

plt.rcParams.update({'font.size': 12})    
# Set the seed
np.random.seed(1)

# Draw 100 items from a two dimensional Gaussian N((1, -1)^{T}, 2I)
X = np.random.multivariate_normal(mean=[1,-1], cov=2*np.eye(2), size=100)

# Plot the scatter plot
plt.scatter(X[:,0], X[:,1])

# Add a title
plt.title("Scatter plot of 100 items from a two dimensional Gaussian")

# Add X_1 and X_2 labels in the plot with latex font
plt.xlabel(r'$X_1$')
plt.ylabel(r'$X_2$')

# Save the plot
plt.savefig("./images/7_a.png")





