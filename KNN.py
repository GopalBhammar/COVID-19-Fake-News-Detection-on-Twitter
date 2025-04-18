#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your precomputed TF-IDF features
dbfile = open('Train', 'rb')
Train_idf = pickle.load(dbfile)
dbfile = open('Validate', 'rb')
Validate_idf = pickle.load(dbfile)

# Assuming you have labels corresponding to the TF-IDF features
# Replace this with your actual labels
y_train = pd.read_csv("train_data.csv")['y']
y_val = pd.read_csv("validate_data.csv")['y']

n_neighbors = 7
metric = 'euclidean'

# Create K-nearest Neighbor classifier with fixed hyperparameters
knn = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)

# Train the model on the training set
knn.fit(Train_idf, y_train)

# Evaluate the model on the validation set
y_val_pred = knn.predict(Validate_idf)

# Print accuracy and classification report for validation set
accuracy_val = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", accuracy_val)
print("Validation Classification Report:\n", classification_report(y_val, y_val_pred))


# In[ ]:
# Display the DataFrame with TF-IDF vector representations
dbfile = open('KNN', 'ab')
pickle.dump(knn, dbfile)
dbfile.close()

