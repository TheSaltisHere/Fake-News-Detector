import numpy as np
import pandas as pd
import itertools

from sklearn.model_selection import train_test_split

from sklearn.linear_model import PassiveAggressiveClassifier

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn.feature_extraction.text import TfidfVectorizer


# Read the data

dataframe = pd.read_csv("dataset.csv")


dataframe.shape
dataframe.head()


labels = dataframe.label

labels.head()


# Split the data into training and test sets

train_x, test_x, train_y, test_y = train_test_split(

    dataframe['text'], labels, test_size=0.2, random_state=42)


vectorizer = TfidfVectorizer()

train_x_dtm = vectorizer.fit_transform(train_x)

test_x_dtm = vectorizer.transform(test_x)


# Initialize a PassiveAggressiveClassifier

clf = PassiveAggressiveClassifier(max_iter=50)

clf.fit(train_x_dtm, train_y)

predicted_y = clf.predict(test_x_dtm)

score = accuracy_score(test_y, predicted_y)

print("Accuracy =", score*100, "%")


# confusion matrix for no of true positives, false positives, true negatives and false negatives

confusion_matrix(test_y, predicted_y, labels=['FAKE', 'REAL'])
