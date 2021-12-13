import json
import urllib
import numpy as np
from flask import Flask
from flask_cors import CORS
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# Exponential Function :
def expo_func(x, a, b):
    return a * b ** x


# Linear Function:
def estimate_coef(x, y):
    # number of observations/points
    n = np.size(x)

    # mean of x and y vector
    m_x = np.mean(x)
    m_y = np.mean(y)

    # calculating cross-deviation and deviation about x
    SS_xy = np.sum(y * x) - n * m_y * m_x
    SS_xx = np.sum(x * x) - n * m_x * m_x

    # calculating regression coefficients
    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x

    return (b_0, b_1)


app = Flask(__name__)
CORS(app)


# Members API Route

@app.route("/predictions/linear")
def linear():
    # get data from json
    cazuri_array = []
    index_array = []
    with urllib.request.urlopen("https://covid19.primariatm.ro/istoric-covid19-tm.json") as url:
        data = json.loads(url.read().decode())
        for i in data:
            cazuri_array.append(i['cazuri'])
            index_array.append(len(cazuri_array) - 1)
    cazuri_array.reverse()
    # print(cazuri_array)
    # print(index_array)

    # observations / data
    y = np.array(cazuri_array)
    x = np.array(index_array)

    # estimating coefficients
    b = estimate_coef(x, y)

    return {
        'aValue': b[0],
        'bValue': b[1]
    }


@app.route("/predictions/log")
def log():
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

    # incat uneroi avem zile cu 0 cazuri trebuie sa facem replace missing values (nu putem aveam log(0)! ->
    # -> putem sa inclocuim valoarea 0 cu media sau cu 1 (basically nu difera valoarea finala)
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
    # Reshaping X :
    X = np.reshape(X, (n, 1))
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

    # Check the accuracy:
    Accuracy = r2_score(Y, Y_plot)
    return {
        'aValue': a,
        'bValue': b,
        'accuracy': Accuracy
    }


@app.route("/predictions/expo")
def expo():
    # Dataset values :
    cazuri_array = []
    index_array = []
    with urllib.request.urlopen("https://covid19.primariatm.ro/istoric-covid19-tm.json") as url:
        data = json.loads(url.read().decode())
        '''data = data[0:200]'''
        for i in data:
            cazuri_array.append(i['cazuri'])
            index_array.append(len(cazuri_array) - 1)
    cazuri_array.reverse()

    # popt :Optimal values for the parameters
    # pcov :The estimated covariance of popt
    popt, pcov = curve_fit(expo_func, index_array, cazuri_array)
    cazuri_pred = expo_func(index_array, popt[0], popt[1])

    # Equation
    a = popt[0].round(4)
    b = popt[1].round(4)

    return {
        'aValue': a,
        'bValue': b
    }


if __name__ == "__main__":
    app.run(debug=True)
