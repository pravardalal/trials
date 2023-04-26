import numpy as np
import matplotlib.pyplot as plt

# Generate the data
mean1, mean2, mean3 = 0, 5, -5
std1, std2, std3 = 1, 2, 3

data1 = np.random.normal(mean1, std1, 1000)
data2 = np.random.normal(mean2, std2, 1000)
data3 = np.random.normal(mean3, std3, 1000)

# Plot the data
fig, ax = plt.subplots()

ax.hist(data1, bins=50, alpha=0.5, label='Data 1')
ax.hist(data2, bins=50, alpha=0.5, label='Data 2')
ax.hist(data3, bins=50, alpha=0.5, label='Data 3')

ax.legend(loc='upper right')
ax.set_title('Gaussian Distribution Samples')
ax.set_xlabel('Value')
ax.set_ylabel('Frequency')

plt.show()
