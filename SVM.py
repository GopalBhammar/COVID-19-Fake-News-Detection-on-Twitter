#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load your precomputed TF-IDF features
dbfile = open('Train', 'rb')
Train_idf = pickle.load(dbfile)
dbfile = open('Validate', 'rb')
Validate_idf = pickle.load(dbfile)

y_train = pd.read_csv("train_data.csv")['y']
y_val = pd.read_csv("validate_data.csv")['y']

# Define the parameter grid for GridSearchCV
# Initialize the SVM model
svm_model = SVC()

# Train the model on the training set
svm_model.fit(Train_idf, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(Validate_idf)

# Print accuracy and classification report
accuracy = accuracy_score(y_val, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:\n", classification_report(y_val, y_pred))


# In[ ]:


# Display the DataFrame with TF-IDF vector representations
dbfile = open('SVM', 'ab')
pickle.dump(svm_model, dbfile)
dbfile.close()

