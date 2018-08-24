import matplotlib.pyplot as plt
import math
from matplotlib import style
import numpy as np

style.use('ggplot')

"""
simple code for plotting the Sigmoid function
"""

def sigmoid(x):
    sig_data = []
    for data in x:
        value = 1 / (1 + math.exp(-data))
        sig_data.append(value)

    return sig_data


def plot_sigmoid(x, y):
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plot for Sigmoid Function')
    plt.show()


if __name__ == '__main__':
    x = np.arange(-15, 15, 0.2, dtype=float)
    y = sigmoid(x)
    plot_sigmoid(x, y)




