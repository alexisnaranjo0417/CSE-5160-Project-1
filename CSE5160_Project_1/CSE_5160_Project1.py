#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


#Load dataset
dataset = pd.read_csv('phishing_legit_dataset_KD_10000.csv')
#Prints the size/dimensions of the dataset
print('Data set size: ', dataset.shape, '\n')
#Prints the names of the columns/attributes of the dataset
print('Column Names: \n', dataset.columns.tolist(), '\n')


#Label X and Y
X = dataset['text'].astype(str)
Y = dataset['phishing_type']


#Convert text → numerical features (TF-IDF)
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)


#Split dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X_tfidf, Y, test_size = 0.2, random_state = 42)


#Create logistic regression
model = LogisticRegression()
model.fit(X_train, Y_train)


#Predictions
Y_pred = model.predict(X_test)


#Evaluation
print("\nModel Accuracy:", accuracy_score(Y_test, Y_pred))