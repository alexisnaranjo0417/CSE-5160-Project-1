#Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import re



#Load dataset
dataset = pd.read_csv('Phishing_Email.csv')


#Data Exploration
#Prints the size/dimensions of the dataset
print('Data set size: ', dataset.shape, '\n')
#Prints the names of the columns/attributes of the dataset
print('Column Names: \n', dataset.columns.tolist(), '\n')


#Check for class balance. Part of data exploration
print('Class Distribution: ')
print(dataset['Email Type'].value_counts())


print('\nDataset looks like: ')
print(dataset.head())


#Step 4 Data Cleansing
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


#Logistic Evaluation
print('\nLogistic Regression Classification Report: \n', classification_report(Y_test, Y_pred))


#SVM model
svm_model= SVC(kernel='linear', C=1, random_state=42)
svm_model.fit(X_train, Y_train)
#Predictions
Y_svm_pred = svm_model.predict(X_test)
#SVM Evaluation
print('\nSVM Classification Report: \n', classification_report(Y_test, Y_svm_pred))


#SVM Hyperparameter Tuning with GridSearchCV
#Define the hyperparameters we want to tune
#param_grid = {'C' : [0.1, 1, 10, 100],
#             'kernel' : ['linear', 'rbf'],
 #            'gamma' : [1, 0.1, 0.01, 0.001, 0.0001]}


#grid = GridSearchCV(SVC(random_state=42), param_grid, refit=True, verbose=3)
#grid.fit(X_train, Y_train)


#print('\nBest Hyperparameters for SVM: ', grid.best_params_)


#Use the best model from grid search to make predictions
#best_svm_model = grid.best_estimator_
#Y_best_svm_pred = best_svm_model.predict(X_test)


#print('\nGRidSearch SVM Classification Report for Best Model: \n', classification_report(Y_test, Y_best_svm_pred))


#Grouped Bar Chart (Comparing All Models)
#Data to plot
labels = ['Logistic Regression', 'SVM']
phishing_f1 = []
safe_email_f1 = []


#Get F1 scores
#Logistic Regression
lr_report = classification_report(Y_test, Y_pred, output_dict = True)
phishing_f1.append(lr_report['0']['f1-score'])
safe_email_f1.append(lr_report['1']['f1-score'])


#SVM
svm_report = classification_report(Y_test, Y_svm_pred, output_dict = True)
phishing_f1.append(svm_report['0']['f1-score'])
safe_email_f1.append(svm_report['1']['f1-score'])


x = np.arange(len(labels)) #Labels locations
width = 0.35 #width of bars


fig, ax = plt.subplots(figsize = (10, 6))
rects1 = ax.bar(x - width/2, phishing_f1, width, label = 'Phishing Email (Class 0)')
rects2 = ax.bar(x + width/2, safe_email_f1, width, label = 'Safe Email (Class 1)')


#Add text for labels, title and axes ticks
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score Comparison')  #by model and email type
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.bar_label(rects1, padding = 3, fmt = '%.2f')
ax.bar_label(rects2, padding = 3, fmt = '%.2f')
ax.set_ylim(0, 1.1) #Set y-axis limit


fig.tight_layout()
plt.show()


#Confusion Matrix (Analyzing your best Model)


#Create a confusion matrix for your best model (SVM)
print('\nSVM Confusion Matrix: ')
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_estimator(svm_model, X_test, Y_test, ax=ax, cmap=plt.cm.Blues, display_labels=le.classes_)    #Shows original labels "Phishing Email", "Safe Email"
ax.set_title('Confusion Matrix for SVM')
plt.show()