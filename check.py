import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
df_pred = pd.read_csv("23CS60R35.csv")
df_original = pd.read_csv("23CS60R35_2.csv")
accuracy_val = accuracy_score(df_original['label'], df_pred["label"])
print("Test Accuracy:", accuracy_val)
