import numpy as np
import sys
import matplotlib.pyplot as plt
np.set_printoptions(threshold = sys.maxsize)

# Playground to experiment with NumPy and its capabilities

# Load data from csv
data = np.genfromtxt("numpydata.csv", delimiter=",", dtype = np.int64)
print(data)

# Arithmetic operations
added_array = 10 * np.ones(data.shape)
subtracted_array = 7 * np.ones(data.shape)
multiplied_array = 3 * np.ones(data.shape)
divided_array = 9 * np.ones(data.shape)
#print(data + added_array)
#print(data - subtracted_array)
#rint(data * multiplied_array)
#print(data / divided_array)

# Array operations
print(np.sin(data))
print(np.cos(data))
print(np.exp(data))
print(data.ravel())

# Universal Functions (ufuncs)
print(np.add.reduce(data, axis = 1))
print(np.bitwise_xor.accumulate(data, axis = 0))

# Filtered arrays
print(data[data > 7])
print(data[data < 4])
print(data[data == 4])
print(data[data % 2 == 0])
print(data[(data > 2) & (data < 5)])
print(data[(data == 2) | (data > 7)])

# Calculated values
print(data.min())
print(data.max())
print(data.mean())
print(data.var())
print(data.std())
print(data.sum())

# Iteration
print("C Order:")
for i in np.nditer(data, order = 'C'):
    print(i)
print("Fortran Order:")
for i in np.nditer(data, order = 'F'):
    print(i)

# Ranges, linspaces and slicing
indexes = np.arange(0, 7, 2)
print(data[..., indexes])
print(data[indexes, :])
circle_angles = np.linspace(0, 2*np.pi, 1000)
print(np.sin(circle_angles))
print(data[2:9, 3:6])

# Stacking
print(np.vstack((np.ones(data.shape), data, np.ones(data.shape))))
print(np.hstack((np.zeros(data.shape), data, np.zeros(data.shape))))

# Splitting
print(np.split(data, [2, 5, 7], axis=1))
print(np.split(data, [4, 10, 17], axis=0))

# Flipping
print(np.flip(data, axis = 0))
print(np.flip(data, axis = 1))
print(np.flip(data))

# Stats operations
counts, bins = np.histogram(data)
print(counts)
_ = plt.hist(bins[:-1], bins, weights=counts)
plt.title("Test Histogram")
plt.show()
bins = [0.0, 2.0, 3.5, 5.5, 8.0, 9.0, 10.5]
print(np.digitize(data, bins))
print(f"Mean : {data.mean()}, Std : {data.std()}")