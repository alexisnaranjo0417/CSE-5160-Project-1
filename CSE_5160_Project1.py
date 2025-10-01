#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import re


#Load dataset
dataset = pd.read_csv('Phishing_Email.csv')
#Prints the size/dimensions of the dataset
print('Data set size: ', dataset.shape, '\n')
#Prints the names of the columns/attributes of the dataset
print('Column Names: \n', dataset.columns.tolist(), '\n')

#Clean dataset
def clean_text(text):
    text = text.lower()  # lowercase
    text = re.sub(r"http\S+", "", text)  # remove URLs
    text = re.sub(r"\S+@\S+", "", text)  # remove email addresses
    text = re.sub(r"\d+", "", text)  # remove numbers
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # remove extra spaces
    return text


#Label X and Y
X = dataset['Email Text'].astype(str).apply(clean_text)
Y = dataset['Email Type']

#Encode labels to numeric
le = LabelEncoder()
Y = le.fit_transform(Y)

#Convert text to numerical features
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

#Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, Y, test_size = 0.2, random_state = 42)

#Create logistic regression
model = LogisticRegression()
model.fit(X_train, Y_train)

#Predictions
Y_pred = model.predict(X_test)

#Logistic evaluation
print("\nModel Accuracy:", accuracy_score(Y_test, Y_pred))

#SVM model
svm_model= SVC(kernel='rbf', C=1, random_state=42)
svm_model.fit(X_train, Y_train)

#Predictions
Y_svm_pred = svm_model.predict(X_test)

#SVM evaluation
print("\nSVM Accuracy:", accuracy_score(Y_test, Y_svm_pred))
