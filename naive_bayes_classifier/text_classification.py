import seaborn as sns
import matplotlib.pyplot as plt
# This will download the data-set if not existing
from sklearn.datasets import fetch_20newsgroups
# Gives weights according to the different words which used to describe the article.Since a document can identify using
# some key words, it can be used those as features to identify the correct document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
# This is useful as there is often a fixed sequence of steps in processing the data, for example feature selection,
# normalization and classification
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix


sns.set()
data = fetch_20newsgroups()
labels = data.target_names
#print(labels)

# Get the training data of the data-set
train_data_set = fetch_20newsgroups(subset='train', categories=labels)
# Get the test data of the data-set
test_data_set = fetch_20newsgroups(subset='test', categories=labels)
#print(train_data_set.data[10])
print(train_data_set.target)
# Creates the model using the multinomial naive-bayes
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# Training the model
model.fit(train_data_set.data, train_data_set.target)
# Predict the label for test data
predicted_labels = model.predict(test_data_set.data)

# Check the accuracy of the model using the confusion matrix. This can be visually illustrates using the heat map
conf_mat = confusion_matrix(test_data_set.target, predicted_labels)
sns.heatmap(conf_mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=train_data_set.target_names,
            yticklabels=train_data_set.target_names)

# Ploting the heat map for the confusion matrix
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.tight_layout(rect=[0, 0, 1, 1])
plt.show()

def predict_labels(s, train=train_data_set, model=model):
    prediction = model.predict([s])
    return train.target_names[prediction[0]]

predicted_label = predict_labels('tree flower plant')
print("Predicted category is", predicted_label)
