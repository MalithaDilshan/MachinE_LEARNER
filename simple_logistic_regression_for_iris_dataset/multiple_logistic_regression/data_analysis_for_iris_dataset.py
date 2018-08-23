import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import  train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


df = pd.read_csv('iris_data.csv')

# print(df.head(5))
'''
   sepal_length  sepal_width  petal_length  petal_width        class
0           5.1          3.5           1.4          0.2  Iris-setosa
1           4.9          3.0           1.4          0.2  Iris-setosa
2           4.7          3.2           1.3          0.2  Iris-setosa
3           4.6          3.1           1.5          0.2  Iris-setosa
4           5.0          3.6           1.4          0.2  Iris-setosa
'''

# print(df.describe())  ## important statistics of data
'''
        sepal_length  sepal_width  petal_length  petal_width
count    150.000000   150.000000    150.000000   150.000000
mean       5.843333     3.054000      3.758667     1.198667
std        0.828066     0.433594      1.764420     0.763161
min        4.300000     2.000000      1.000000     0.100000
25%        5.100000     2.800000      1.600000     0.300000
50%        5.800000     3.000000      4.350000     1.300000
75%        6.400000     3.300000      5.100000     1.800000
max        7.900000     4.400000      6.900000     2.500000
'''

# print(df.info())  ## check for null values
'''
# there are no any null values of the data frame

RangeIndex: 150 entries, 0 to 149
Data columns (total 5 columns):
sepal_length_in_cm    150 non-null float64
sepal_width_in_cm     150 non-null float64
patel_length_in_cm    150 non-null float64
petal_width_in_cm     150 non-null float64
class                 150 non-null object
dtypes: float64(4), object(1)
memory usage: 5.9+ KB
'''

# print(df['class'].value_counts())
'''
Iris-setosa        50
Iris-virginica     50
Iris-versicolor    50
Name: class, dtype: int64
'''

## drop the 'Iris-virginica' class and use logistic regression for 'Iris-setosa' and 'Iris-versicolor'
df_new = df[df['class'] != 'Iris-virginica']

## histograms
# df_new.hist(column='petal_width', bins=25, figsize=(10, 15))

## remove outliers
# there is a one outlier in the 'Iris-setosa' class
df_new = df_new.drop(df_new[(df_new['sepal_width'] < 2.5) & (df_new['class'] == "Iris-setosa")].index)

sns.pairplot(df_new, hue='class', size=2.5)   ## plot whole numerical values in the data frame over class
# sns.swarmplot(x="class", y="petal_length", data=df)  ## plot over two variables
# plt.show()

## binary encoding
df_new['class'].replace(['Iris-setosa', 'Iris-versicolor'], [0, 1], inplace=True)

# print(df_new.head(5))
'''
    sepal_length  sepal_width  petal_length  petal_width  class
0           5.1          3.5           1.4          0.2      0
1           4.9          3.0           1.4          0.2      0
2           4.7          3.2           1.3          0.2      0
3           4.6          3.1           1.5          0.2      0
4           5.0          3.6           1.4          0.2      0
'''

## model construction for logistic regression
X = np.array(df_new.drop(df_new.columns[[4]], axis=1))  ## shape 99x4
y = np.array(df_new.drop(df_new.columns[[0, 1, 2, 3]], axis=1))  ## shape 99x1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## after running one time this block can be commented and loaded pickle file can be used instead
# -----------------------------------------------------------------------------------------------------
clf = LogisticRegression()
clf.fit(X_train, y_train.ravel())

## dump the classifier in to the pickle file. this will save the time for classification
with open('logistic_regression.pickle', 'wb') as f:
    pickle.dump(clf, f)
# -----------------------------------------------------------------------------------------------------
pickle_file = open('logistic_regression.pickle', 'rb')
clf = pickle.load(pickle_file)

print("'Intercept' and 'coefficients'", clf.intercept_, clf.coef_)

y_predicted = clf.predict(X_test)
# accuracy = clf.score(X_test, y_test)
accuracy = accuracy_score(y_predicted.T, y_test)
print(accuracy)
