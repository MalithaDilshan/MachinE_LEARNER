# Simple program for manually calculating the standard deviation and likelihood for a numerical value.
# There are python inbuilt functions for calculating mean and std (using numpy.std())

import math


# calculate mean
def calculate_mean(dataset):
    sum = 0
    length = len(dataset)
    for i in range(0, length):
        sum += dataset[i]
    mean = sum/length

    return mean


# calculate std
def calculate_std(mean, dataset):
    length = len(dataset)
    squared_sum = 0
    for data in dataset:
        squared_sum += (data-mean)**2
    std = math.sqrt(squared_sum/(length-1))

    return std


# calculate likelihood
def calculate_likelihood(data, std, mean):
    power = -(((data - mean)**2)/(2*(std)**2))
    value_exp = math.exp(power)
    divider = std*math.sqrt(2*math.pi)

    return value_exp/divider

# edit here to input dataset as an array and number
#dataset = [32, 35, 25, 36, 29, 27, 38,33, 34, 39, 36, 29, 23]
dataset = [25, 23, 33, 35, 28, 26, 28, 33, 40]
num = 33

mean = calculate_mean(dataset)
std = calculate_std(mean, dataset)
print("mean is :", mean, "standard deviation", std)
likelihood = calculate_likelihood(num, std, mean)
print("likelihood of", num, ":", likelihood)
