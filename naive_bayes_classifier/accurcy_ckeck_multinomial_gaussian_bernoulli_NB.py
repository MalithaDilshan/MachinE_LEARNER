# Compare the accuracy of three main categories of Naive Bayes Classifier. Here the iris data-set has used to build
# these models. Since the iris data-set includes continues data, the Gaussian Naive Bayes should give accurate results
# when compare with others. To illustrate the results heat map has used from the seaborn package.

from sklearn import datasets
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

sns.set()

iris = datasets.load_iris()
X = iris.data
y = iris.target

# training the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# plotting the heat map for the confusion matrix
def plot_me(title):
    plt.title(title)
    plt.xlabel('Actual Labels')
    plt.ylabel('Predicted Labels')
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.show()


def multinomial_NB():
    multinomial_model = MultinomialNB()
    multinomial_model.fit(X_train, y_train)
    predicted_labels_multinomial = multinomial_model.predict(X_test)

    conf_mat = confusion_matrix(y_test, predicted_labels_multinomial)
    sns.heatmap(conf_mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=iris.target_names,
    yticklabels=iris.target_names)
    plot_me("Plot for Multinomial Naive Bayes")


def gaussian_NB():
    gaussian_model = GaussianNB()
    gaussian_model.fit(X_train, y_train)
    predicted_labels_multinomial = gaussian_model.predict(X_test)

    conf_mat = confusion_matrix(y_test, predicted_labels_multinomial)
    sns.heatmap(conf_mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=iris.target_names,
    yticklabels=iris.target_names)
    plot_me("Plot for Gaussian Naive Bayes")


def bernoulli_NB():
    bernoulli_model = BernoulliNB(binarize=True)
    bernoulli_model.fit(X_train, y_train)
    predicted_labels_multinomial = bernoulli_model.predict(X_test)

    conf_mat = confusion_matrix(y_test, predicted_labels_multinomial)
    sns.heatmap(conf_mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=iris.target_names,
                yticklabels=iris.target_names)
    plot_me("Plot for Bernoulli Naive Bayes")

# get one by one plot for confusion matrices
multinomial_NB()
gaussian_NB()
bernoulli_NB()

