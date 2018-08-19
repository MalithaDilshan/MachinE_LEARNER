import numpy as np
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('ggplot')

'''
implement the best fit line (simple linear regression learning model) using the Gradient Decent optimizing algorithm
'''

# static data - positive correlated data set (instead of using random data set)
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
    # X = [i for i in range(len(Ys))]
    # plt.plot(X, Y_mean, label='Mean Line')
    squared_error_reg_line = squared_error(Ys, Y_line)
    squared_error_Y_mean = squared_error(Ys, Y_mean)
    if squared_error_Y_mean != 0:
        r = 1 - (squared_error_reg_line/squared_error_Y_mean)
    return r


# get the slope and intercept using the statistic equations
def Gradient_Decent(Xs, Ys, m_current=0, b_current=0, learning_rate=0.0001):
    # variables
    N = float(len(Ys))
    precision = 0.0000001
    step_size = 1
    iter1 = 0
    cost  = -1
    while step_size > precision:
        y_current = (m_current * Xs) + b_current
        cost = sum([data**2 for data in(Ys - y_current)])/N
        m_gradient = -(2/N) * sum(Xs * (Ys - y_current))
        b_gradient = -(2/N) * sum(Ys - y_current)
        m_previous = m_current
        # apply gradient decent
        m_current -= learning_rate * m_gradient
        b_current -= learning_rate * b_gradient
        step_size = abs(m_current - m_previous)
        iter1 += 1
        print("** ", m_current)

    print("Number of Iterations:", iter1)
    print("Final MSE:", cost)
    return m_current, b_current


# illustate the linear model with data
def plot(Y_reg_line, Xs, Ys):
    plt.scatter(Xs, Ys, s=10, color='b')
    plt.plot(Xs, Y_reg_line, label='Regression Line')
    # plt.legend()
    plt.show()


# data set
def data_set(length, variance, step=1, correlation=False):
    current_value = 1.5
    Ys = []
    for i in range(0, length):
        Y_new = current_value + random.randrange(-variance, variance)
        Ys.append(Y_new)
        if correlation and correlation == 'p':
            current_value += step
        elif correlation and correlation == 'n':
            current_value -= step
    Xs = [i for i in range(len(Ys))]
    return np.array(Xs, dtype=np.float64), np.array(Ys, dtype=np.float64)


# main function
if __name__ == "__main__":

    '''
    Modify this to use random or static data
    random data set - invoke the function 'data_set(number_of_points, variance, step_size, correlation_type ['n'- for
    negative, 'p' for positive, otherwise keep default as 'False'])'. As the minimum requirement feed number_of_points,
    variance and step_size.
    '''

    # random data set
    Xs, Ys = data_set(80, 5, 1, 'p')
    # static data set
    # Xs, Ys = X_static, Y_static
    m, b = Gradient_Decent(Xs, Ys, 0, 0, 0.0001)
    print(m, b)
    Y_line = [(m*x)+b for x in Xs]
    # get the R-Squared error
    r_squared_error = coefficient_of_determination(Ys, Y_line)
    print("R_Squared_Error:", r_squared_error)
    # plotting
    plot(Y_line, Xs, Ys)