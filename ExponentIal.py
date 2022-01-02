# Import required libraries:
import urllib, json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Dataset values :
# day = np.arange(0, 8)
# weight = np.array([251, 209, 157, 129, 103, 81, 66, 49])
# get data from json
cazuri_array = []
index_array = []
with urllib.request.urlopen("https://covid19.primariatm.ro/istoric-covid19-tm.json") as url:
    data = json.loads(url.read().decode())
    # data = data[0:200] vedem ca in ultimele 200 de zile se observa o crestere, dar daca lucram pe tot setul, va fi o scadere
    for i in data:
        cazuri_array.append(i['cazuri'])
        index_array.append(len(cazuri_array) - 1)
cazuri_array.reverse()


# Exponential Function :
def expo_func(x, a, b):
    return a * b ** x


# popt :Optimal values for the parameters
# pcov :The estimated covariance of popt
popt, pcov = curve_fit(expo_func, index_array, cazuri_array)
cazuri_pred = expo_func(index_array, popt[0], popt[1])
# Plotting the data
plt.plot(index_array, cazuri_pred, 'r-')
plt.scatter(index_array, cazuri_array, label='Day vs Weight')
plt.title("Day vs Cases a*b^x")
plt.xlabel('Day')
plt.ylabel('Cases')
plt.legend()
plt.show()
# Equation
a = popt[0].round(4)
b = popt[1].round(4)
print(f'The equation of regression line is y={a}*{b}^x')

# save output in a  txt file
with open("output.txt", "a") as f:
    print(f'y={a}*{b}^x', file=f)

# save output in a json file
plot = {
    'a': a,
    'b': b
}
with open('plot.json', 'w') as json_file:
    json.dump(plot, json_file)