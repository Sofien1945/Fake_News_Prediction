""""
Fake news prediction using passive agressive classifier
Part of the https://data-flair.training/ machine learning course
Date:27.10.2021
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

#Read + Split datasets to features and labels
file_path = 'C:/Users/Sofien/Desktop/news.csv'
df = pd.read_csv(file_path)
X = df.iloc[:,1]
y = df.iloc[:,-1]

#Slit the train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

#Apply TFID Vertorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#Apply the passive agressive classifier
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(X_train_vec, y_train)
y_pred = pac.predict(X_test_vec)
score = accuracy_score(y_test, y_pred)
print(f"Accuracy score: {score}")
cf = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(cf, annot=True, fmt=".2f", linewidths=.5, square = True, cmap = 'Blues_r',xticklabels=['REAL','FAKE'],yticklabels=['REAL','FAKE'],)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()