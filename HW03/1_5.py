import matplotlib.pyplot as plt
import numpy as np


tpr = np.array([0, 2/6, 4/6, 6/6, 6/6])

fpr = np.array([0, 0, 1/4, 2/4, 4/4])


plt.plot(fpr, tpr, 'b')
plt.xlabel('False Positive Rate')
# start x at 0
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.savefig('./images/1_5.png')
plt.show()
