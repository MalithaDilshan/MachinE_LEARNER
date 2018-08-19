import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import pickle

import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')


'''
A feature is one column of the data in your input set. For instance,
if you're trying to predict the type of pet someone will choose, your
input features might include age, home region, family income, etc. The
label is the final choice, such as dog, fish, iguana, rock, etc.
Once you've trained your model, you will give it sets of new input
containing those features; it will return the predicted "label"
(pet type) for that person.

before shift
Adj. Close    HL_PCT  PCT_Change  Adj. Volume
Date
2004-08-19   50.322842  3.712563    0.324968   44659000.0
2004-08-20   54.322689  0.710922    7.227007   22834300.0
2004-08-23   54.869377  3.729433   -1.227880   18256100.0
2004-08-24   52.597363  6.417469   -5.726357   15247300.0
2004-08-25   53.164113  1.886792    1.183658    9188600.0

after shift

Adj. Close    HL_PCT  PCT_Change  Adj. Volume      label
Date
2004-08-19   50.322842  3.712563    0.324968   44659000.0  54.322689
2004-08-20   54.322689  0.710922    7.227007   22834300.0  54.869377
2004-08-23   54.869377  3.729433   -1.227880   18256100.0  52.597363
2004-08-24   52.597363  6.417469   -5.726357   15247300.0  53.164113
2004-08-25   53.164113  1.886792    1.183658    9188600.0  54.122070


Original tail(5) of the data-frame

               Open     High      Low    Close     Volume  Ex-Dividend  \
Date
2018-03-21  1092.57  1108.70  1087.21  1094.00  1990515.0          0.0
2018-03-22  1080.01  1083.92  1049.64  1053.15  3418154.0          0.0
2018-03-23  1051.37  1066.78  1024.87  1026.55  2413517.0          0.0
2018-03-26  1050.60  1059.27  1010.58  1054.09  3272409.0          0.0
2018-03-27  1063.90  1064.54   997.62  1006.94  2940957.0          0.0

            Split Ratio  Adj. Open  Adj. High  Adj. Low  Adj. Close  \
Date
2018-03-21          1.0    1092.57    1108.70   1087.21     1094.00
2018-03-22          1.0    1080.01    1083.92   1049.64     1053.15
2018-03-23          1.0    1051.37    1066.78   1024.87     1026.55
2018-03-26          1.0    1050.60    1059.27   1010.58     1054.09
2018-03-27          1.0    1063.90    1064.54    997.62     1006.94

            Adj. Volume
Date
2018-03-21    1990515.0
2018-03-22    3418154.0
2018-03-23    2413517.0
2018-03-26    3272409.0
2018-03-27    2940957.0
'''

# Data pre-processing
df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])*100/(df['Adj. Close'])
df['PCT_Change'] = (df['Adj. Close'] - df['Adj. Open'])*100/(df['Adj. Open'])

df_old = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]

df = df[['Adj. Close', 'HL_PCT', 'PCT_Change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
# fill the NAN to a number
df.fillna(-11111)
# math.ceil is responsible for round up the decimal value to the upper nearest int
forecast_out = int(math.ceil(0.01*len(df)))
df['Label'] = df[forecast_col].shift(-forecast_out)

X = np.array(df.drop(['Label'], 1))
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]  # get the last 35 rows
X = X[:-forecast_out]  # remove the last 35 rows

'''
The issue with sparsity is that it very biased or in statistical terms skewed. So,
therefore, scaling the data brings all your values onto one scale eliminating the
sparsity. In regards to know how it works in mathematical detail, this follows the
same concept of Normalization and Standardization.
'''
# remove the NAN values (entire row) in the label column after shifting
df.dropna(inplace=True)
'''
After dropping

            Adj. Close    HL_PCT  PCT_Change  Adj. Volume    Label
Date
2018-01-30     1177.37  0.896914   -0.029718    1792602.0  1094.00
2018-01-31     1182.22  0.346805   -0.134312    1643877.0  1053.15
2018-02-01     1181.59  0.495942    0.476195    2774967.0  1026.55
2018-02-02     1119.20  1.081129   -0.729098    5798880.0  1054.09
2018-02-05     1068.76  4.325574   -2.893850    3742469.0  1006.94
'''

y = np.array(df['Label'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.33)

# after running one time this block can be commented and loaded pickle file can be used instead
# -----------------------------------------------------------------------------------------------------
clf = LinearRegression(n_jobs=-1)

clf.fit(X_train, y_train)
# dump the classifier in to the pickle file. this will save the time for classification
with open('linear_regression.pickle', 'wb') as f:
    pickle.dump(clf, f)
# -----------------------------------------------------------------------------------------------------

# loading the created pickle file
pickle_file = open('linear_regression.pickle', 'rb')
clf = pickle.load(pickle_file)

accuracy = clf.score(X_test, y_test)

# predict the last 35 values in the 'Adj. Close' column using the trained classifier
forecast_set = clf.predict(X_lately)
df_old['Forecast'] = np.nan

'''
Data frame
            Adj. Close    HL_PCT  PCT_Change  Adj. Volume      Label  Forecast
Date
2004-08-19   50.322842  3.712563    0.324968   44659000.0  69.078238      NaN
2004-08-20   54.322689  0.710922    7.227007   22834300.0  67.839414      NaN
2004-08-23   54.869377  3.729433   -1.227880   18256100.0  68.912727      NaN
2004-08-24   52.597363  6.417469   -5.726357   15247300.0  70.668146      NaN
2004-08-25   53.164113  1.886792    1.183658    9188600.0  71.219849      NaN
'''

last_date = df_old.iloc[-1].name  # last row of the data frame
last_unix = last_date.timestamp()  # get the time using the seconds
one_day = 86400.0
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)  # get the next date
    next_unix += one_day
    # put the predicted values with the dates
    df_old.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(accuracy)
# plotting the actual data and the
df_old['Adj. Close'].plot()
df_old['Forecast'].plot()
plt.legend(loc=2)
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.show()


