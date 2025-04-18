import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
import fasttext
import subprocess


models_names = [
    "KNN",
    "Logistic",
    "Neural",
    "FastText",
    "kMeans",
    "SVM"
]
# Print the list of models
# print(models[int(sys.argv[1])])
for model in models_names:
    print("Model = ",model)
    # subprocess.run(['python3', model+'.py'])
    if model=="FastText":
        models = fasttext.load_model('FastText.bin')
    # Load your precomputed TF-IDF features
    else:
        dbfile = open(model, 'rb')
        models = pickle.load(dbfile)
    dbfile = open('Test', 'rb')
    test_data = pickle.load(dbfile)
    dbfile = open('Test', 'rb')
    y_test = pd.read_csv("test_data.csv")
    
    # Make predictions on the test set
    if model=="FastText":
        y_pred = models.predict(y_test['X'].tolist())
        y_val_pred_labels = [label[0].replace('__label__', '') for label in y_pred[0]]
        conf_matrix = confusion_matrix(y_test['y'].tolist(), y_val_pred_labels)
        print("Confusion Matrix:")
        print(conf_matrix)
        # Print accuracy and classification report for validation set
        accuracy_val = accuracy_score(y_test['y'].tolist(), y_val_pred_labels)
        print("Test Accuracy:", accuracy_val)
        print("Test Classification Report:\n", classification_report(y_test['y'].tolist(), y_val_pred_labels))
    # elif model=="kMeans":
    #     train_clusters = models.predict(test_data)
    
    #     # Create a new DataFrame with cluster assignments
    #     cluster_df_train = pd.DataFrame({'cluster': train_clusters, 'y': y_test['y']})
    #     # Map cluster labels to the most frequent true label in each cluster
    #     cluster_label_mapping = cluster_df_train.groupby('cluster')['y'].agg(lambda x: x.value_counts().index[0]).to_dict()
    #     cluster_df_train['predicted_label'] = cluster_df_train['cluster'].map(cluster_label_mapping)
    #     conf_matrix = confusion_matrix(cluster_df_train['y'], cluster_df_train['predicted_label'])
    #     print("Confusion Matrix:")
    #     print(conf_matrix)
    #     # Print accuracy and classification report for validation set
    #     accuracy_val = accuracy_score(cluster_df_train['y'], cluster_df_train['predicted_label'])
    #     print("Test Accuracy:", accuracy_val)
    #     print("Test Classification Report:\n", classification_report(cluster_df_train['y'], cluster_df_train['predicted_label']))
    else:
        y_pred = models.predict(test_data)
        # conf_matrix = confusion_matrix(y_test['y'], y_pred)
        print("Confusion Matrix:")
        # print(conf_matrix)
        # Print accuracy and classification report for validation set
        # accuracy_val = accuracy_score(y_test['y'], y_pred)
        # print("Test Accuracy:", accuracy_val)
        # print("Test Classification Report:\n", classification_report(y_test['y'], y_pred))
        final_df = pd.DataFrame({'label': y_pred})
        final_df.to_csv('23CS60R24.csv', index=False)
