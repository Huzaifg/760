import matplotlib.pyplot as plt
import numpy as np
import math

# A script that plots ln(n) and log2(n) for n = 1,2,...,10000
# Create a list of n values
n = np.arange(1,10001)

# Create a list of ln(n) values
ln_n = np.log(n)

# Create a list of log2(n) values
log2_n = np.log2(n)

# Plot ln(n) and log2(n) against n
plt.plot(n, ln_n, label = "ln(n)")
plt.plot(n, log2_n, label = "log2(n)")
# Add a legend
plt.legend()
# Add a title
plt.title("ln(n) and log2(n) against n")
# Add x and y labels
plt.xlabel("n")
#save the plot
plt.savefig("./images/4_a.png")
# Show the plot
plt.show()
