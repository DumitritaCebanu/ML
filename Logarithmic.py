# Import required libraries:
import json
import urllib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score


# Dataset:
# Y = a + b*ln(X)
cazuri_array = []
index_array = []
with urllib.request.urlopen("https://covid19.primariatm.ro/istoric-covid19-tm.json") as url:
    data = json.loads(url.read().decode())
    # data = data[0:200]
    for i in data:
        cazuri_array.append(i['cazuri'])
        index_array.append(len(cazuri_array) - 1)
cazuri_array.reverse()
cases = cazuri_array

# m = np.median(cases[cases > 0])
# cases[cases == 0] = m

cases = [1 if x == 0 else x for x in cases]

X = np.array(cases)
Y = 10 + 2 * np.log(X)

# Adding some noise to calculate error!
Y_noise = np.random.rand(len(Y))
Y = Y + Y_noise
plt.scatter(X, Y)
# 1st column of our X matrix should be 1:
n = len(X)
x_bias = np.ones((n, 1))
print(X.shape)
print(x_bias.shape)
# Reshaping X :
X = np.reshape(X, (n, 1))
print(X.shape)
# Going with the formula:
# Y = a + b*ln(X)
X_log = np.log(X)
# Append the X_log to X_bias:
x_new = np.append(x_bias, X_log, axis=1)
# Transpose of a matrix:
x_new_transpose = np.transpose(x_new)
# Matrix multiplication:
x_new_transpose_dot_x_new = x_new_transpose.dot(x_new)
# Find inverse:
temp_1 = np.linalg.inv(x_new_transpose_dot_x_new)
# Matrix Multiplication:
temp_2 = x_new_transpose.dot(Y)
# Find the coefficient values:
theta = temp_1.dot(temp_2)
# Plot the data:
a = theta[0]
b = theta[1]
Y_plot = a + b * np.log(X)
plt.scatter(X, Y)
plt.plot(X, Y_plot, c="r")
plt.show()
# Check the accuracy:
Accuracy = r2_score(Y, Y_plot)
print(Accuracy)
print(a, b)

