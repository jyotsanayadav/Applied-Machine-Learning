# 1. NumPy: Create a 1D NumPy array and reshape it into 2D and 3D arrays.
# Task: Create a 1D array with values from 1 to 12.
# Reshape it into: A 2D array with dimensions (3, 4)., and A 3D array with dimensions (2, 2, 3).
import numpy as np
array_1d = np.arange(1, 13)
array_2d = array_1d.reshape(3, 4) 
array_3d = array_1d.reshape(2, 2, 3) 
print("1D Array:\n", array_1d)
print("2D Array (3x4):\n", array_2d)
print("3D Array (2x2x3):\n", array_3d)

# 2. Pandas: Create two Series with custom indices and perform division. 
import pandas as pd
series1 = pd.Series([10, 20, 30], index=['a', 'b', 'c'])
series2 = pd.Series([2, 4, 5], index=['a', 'b', 'c'])
result_series = series1 / series2
print("\nSeries A:\n", series1)
print("Series B:\n", series2)
print("Result Series:\n", result_series)

# 3. Matplotlib: Plot a red line graph for given (x, y) values. 
import matplotlib.pyplot as plt
x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]
plt.plot(x, y, color='red')
plt.title("Red Line Graph")
plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.show()

# 4. SciPy: Compute determinant, eigenvalues, and eigenvectors of a matrix.
import numpy as np
from scipy import linalg
A=np.array([[4, 2], [1, 3]])
det_A = linalg.det(A) 
eigenvals_A = linalg.eigvals(A)
eigenvecs_A = linalg.eig(A)[1]
print("Determinant of A:", det_A)
print("Eigenvalues of A:", eigenvals_A)
print("Eigenvectors of A:\n", eigenvecs_A)

# 5. Statistics: Compute mean, median, mode, variance, and standard deviation.
import numpy as np
from scipy import stats
data=[1, 2, 2, 3, 4, 5, 5, 5, 6]
mean = np.mean(data)
median = np.median(data)
mode = stats.mode(data, keepdims=True)
variance = np.var(data)
std_dev = np.std(data)
print("Mean:", mean)
print("Median:", median)
print("Mode:", mode.mode[0])
print("Variance:", variance)
print("Standard Deviation:", std_dev)