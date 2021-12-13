import numpy as np
import matplotlib.pyplot as plt
import urllib, json


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


def plot_regression_line(x, y, b):
    # plotting the actual points as scatter plot
    plt.scatter(x, y, color="m",
                marker="o", s=30)

    # predicted response vector
    y_pred = b[0] + b[1] * x

    # plotting the regression line
    plt.plot(x, y_pred, color="g")

    # putting labels
    plt.xlabel('x')
    plt.ylabel('y')

    # function to show plot
    plt.show()


def main():
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
    print("Estimated coefficients:\nb_0 = {}  \
          \nb_1 = {}".format(b[0], b[1]))

    # plotting regression line
    plot_regression_line(x, y, b)


if __name__ == "__main__":
    main()



#_______________V2__________________________
#  Import required libraries:
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn import linear_model
# import urllib, json
#
#
# # get data from json
# cazuri_array = []
# data_array = []
# with urllib.request.urlopen("https://covid19.primariatm.ro/istoric-covid19-tm.json") as url:
#     data = json.loads(url.read().decode())
#     for i in data:
#         cazuri_array.append(i['cazuri'])
#         data_array.append((i['data']))
#
# # ENGINESIZE vs CO2EMISSIONS:
# plt.scatter(data_array, cazuri_array, color="blue")
# plt.xlabel("zi")
# plt.ylabel("cazuri")
# plt.show()
# # Generating training and testing data from our data:
# # We are using 80% data for training.
# train = cazuri_array[:(int((len(cazuri_array) * 0.8)))]
# test = cazuri_array[(int((len(cazuri_array) * 0.8))):]
# # Modeling:
# # Using sklearn package to model data :
# regr = linear_model.LinearRegression()
# train_x = np.array(train)
# train_y = np.array(train)
# regr.fit(train_x, train_y)
# # The coefficients:
# print("coefficients : ", regr.coef_)  # Slope
# print("Intercept : ", regr.intercept_)  # Intercept
# # Plotting the regression line:
# plt.scatter(train, train, color='blue')
# plt.plot(train_x, regr.coef_ * train_x + regr.intercept_, '-r')
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
#
#
# # Predicting values:
# # Function for predicting future values :
# def get_regression_predictions(input_features, intercept, slope):
#     predicted_values = input_features * slope + intercept
#     return predicted_values
#
#
# # Predicting emission for future car:
# my_engine_size = 3.5
# estimatd_emission = get_regression_predictions(my_engine_size, regr.intercept_[0], regr.coef_[0][0])
# print("Estimated Emission :", estimatd_emission)
# # Checking various accuracy:
# from sklearn.metrics import r2_score
#
# test_x = np.array(test)
# test_y = np.array(test)
# test_y_ = regr.predict(test_x)
# print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
# print("Mean sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
# print("R2-score: %.2f" % r2_score(test_y_, test_y))
