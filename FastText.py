#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import fasttext
from sklearn.metrics import accuracy_score, classification_report
import pickle
# Assuming you have labels corresponding to the TF-IDF features
# Replace this with your actual labels
df_train = pd.read_csv("train_data.csv")
df_val = pd.read_csv("validate_data.csv")

# Format the training data in the fastText format
train_data = []
for index, row in df_train.iterrows():
    label = f'__label__{row["y"]}'  # Assuming 'y' is the label column
    text = row["X"]  # Assuming 'X' is the text column
    train_data.append(f'{label} {text}')

with open('train.txt', 'w') as f:
    f.write('\n'.join(train_data))

# Train FastText model
model = fasttext.train_supervised(input='train.txt', epoch=25, lr=1.0, wordNgrams=2)

# Evaluate the model on the validation set
y_val_pred = model.predict(df_val['X'].tolist())
y_val_pred_labels = [label[0].replace('__label__', '') for label in y_val_pred[0]]

# Print accuracy and classification report for validation set
accuracy_val = accuracy_score(df_val['y'].tolist(), y_val_pred_labels)
print("Validation Accuracy:", accuracy_val)
print("Validation Classification Report:\n", classification_report(df_val['y'].tolist(), y_val_pred_labels))

# Print accuracy and classification report for validation set
accuracy_val = accuracy_score(df_val['y'].tolist(), y_val_pred_labels)
print("Validation Accuracy:", accuracy_val)
print("Validation Classification Report:\n", classification_report(df_val['y'].tolist(), y_val_pred_labels))


# In[5]:


model.save_model('FastText.bin')


# In[ ]:




