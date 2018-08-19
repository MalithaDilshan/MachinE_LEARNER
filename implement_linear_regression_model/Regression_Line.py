import numpy as np
import math
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('ggplot')

'''
implement the best fit line (simple linear regression learning model) using the statistic equations. Here it can not
be gurrentied to get the better linear model that has the minimum error (least square error)
'''
# static data (instead of using random data set)
X_static = np.array([x for x in range(0, 80)])
Y_static = np.array([38.5, 29.2, -6.1, -0.4, -17.7, 2.0, 30.7, 26.4, 11.1, -11.2, 16.5, 42.2, -30.1, 27.6, 5.3, 18.0,
                     -15.3, 20.4, 19.1, 2.8, 37.5, 36.2, 5.9, -2.4, -1.7, 47.0, 43.7, 28.4, 33.1, 39.8, 24.5, -11.8,
                     19.9, 4.6, -13.7, 48.0, 60.7, 32.4, 47.1, 33.8, 5.5, 46.2, 9.9, 39.6, 24.3, 35.0, 43.7, 29.4, 65.1,
                     53.8, 20.5, 0.2, -0.1, 1.6, 22.3, 76.0, 71.7, 55.4, 11.1, 19.8, 34.5, 63.2, 60.9, 23.6, 16.3, 62.0,
                     62.7, 68.4, 32.1, 39.8, 76.5, 47.2, 55.9, 52.6, 41.3, 85.0, 29.7, 31.4, 16.1, 69.8])


# get the squared error
def squared_error(Ys, Y_line):
    error = sum((Ys - Y_line)**2)
    return error


def coefficient_of_determination(Ys, Y_line):
    Y_mean = [mean(Ys) for y in Ys]
    squared_error_reg_line = squared_error(Ys, Y_line)
    print("Squared_Error:", squared_error_reg_line)
    squared_error_Y_mean = squared_error(Ys, Y_mean)
    if(squared_error_Y_mean != 0):
        r = 1 - (squared_error_reg_line/squared_error_Y_mean)
    return r


# get the slope and intercept using the statistic equations
def line_slope_and_intercept(Xs, Ys):
    slope = ((mean(Xs)*mean(Ys)) - mean(Xs*Ys))/((mean(Xs)*mean(Xs))-mean(Xs*Xs))
    intercept = mean(Ys) - slope*mean(Xs)
    return slope, intercept


# illustate the linear model with data
def plot(Y_reg_line, Xs, Ys):
    plt.scatter(Xs, Ys, s=10, color='b')
    plt.plot(Xs, Y_reg_line)
    plt.show()


# data set
def data_set(length, varience, step=1, correlation=False):
    current_value = 1.5
    Ys = []
    for i in range(0, length):
        Y_new = current_value + random.randrange(-varience, varience)
        Ys.append(Y_new)
        if correlation and correlation == 'p':
            current_value += step
        elif correlation and correlation == 'n':
            current_value -= step
    Xs = [i for i in range(len(Ys))]
    return np.array(Xs, dtype=np.float64), np.array(Ys, dtype=np.float64)


# main function
if __name__ == "__main__":

    # Xs, Ys = data_set(80, 30, 0.7)
    Xs, Ys = X_static, Y_static
    m, b = line_slope_and_intercept(X_static, Y_static)
    print(m, b)
    Y_line = [(m*x)+b for x in X_static]
    # get the R-Squared error
    r_squared_error = coefficient_of_determination(Y_static, Y_line)
    print("R_Squared_Error:", r_squared_error)
    # plotting
    plot(Y_line, X_static, Y_static)


