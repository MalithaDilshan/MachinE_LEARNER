from sklearn import datasets
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

style.use('ggplot')

'''
0 represent 'Setosa' and 'Versicolor' classes and 1 represent 'Virginica' class. This program uses the logistic
regression model to predict the class using the probability by inputting the 'Petal width' as a feature.
'''
# load iris data (this is already preprocessed data set - no missing values or outliers)
iris = datasets.load_iris()
X = []
y = []

original_length = 0
# select the 'patel width' as the feature (this is seem like simple logistic regression)
for data in iris['data']:
    array = np.array(data, dtype=float)
    X.append(array[3])

# create y using the 1 and 0 (for binary logistic regression - as binary classifier)
for data in iris['target']:
    value = -1
    if data == 2:
        value = 1
    else:
        value = 0
    y.append(value)
    original_length += 1

print("Length of the 'y':", original_length)

# training the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
model = LogisticRegression()
model.fit(np.array(X_train).reshape(-1, 1), y_train)
y_prob = model.predict(np.array(X_test).reshape(-1, 1))

# check the accuracy using the numerical data
accuracy = model.score(np.array(X_test).reshape(-1, 1), y_test)
print("Accuracy", accuracy)

# get the optimized coefficient( weights for the feature) and suitable bias for the input function z
# (this part is hidden to the user)
print("'Model bias' and 'Feature coefficients'", model.intercept_, model.coef_)

# plot the predicted class and compare the accuracy using the plots
length = len(np.array(y_test))   # get the length of test set for labels
Xs = [i for i in range(0, length)]
plt.scatter(Xs, y_test, s=15, color='r', label='Original Class')
plt.scatter(Xs, y_prob, s=5, color='b', label='Predicted Class')
plt.title("Original class Vs Predicted class")
plt.ylabel('Class')
plt.legend()
plt.show()


