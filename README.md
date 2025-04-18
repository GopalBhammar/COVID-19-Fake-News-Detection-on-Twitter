# COVID-19-Fake-News-Detection-on-Twitter


# COVID-19 Fake News Detection on Twitter

This project focuses on detecting fake news related to COVID-19 on Twitter using a variety of machine learning and deep learning models. The goal is to classify tweets as either **real** or **fake** using preprocessed and vectorized data.

---

## Models Implemented

| Model      | Validation Accuracy | Test Accuracy | Notes            |
|------------|---------------------|----------------|------------------|
| KNN        | 90.66%              | 92.17%         | Simple, fast     |
| Logistic   | 92.92%              | 92.26%         | Interpretable    |
| Neural Net | 94.43%              | 93.68%         | Strong DL model  |
| FastText   | 94.72%              | 82.00%         | Word embeddings  |
| SVM        | 94.53%              | 94.05%         | High precision   |
| kMeans     | N/A                 | 82.07%         | Unsupervised     |

---

## How to run

    make run

This will:

    Run each model script
    
    Load the model (from pickle or .bin)
    
    Predict on test data
    
    Print:
    
      Confusion Matrix
      
      Accuracy
      
      Classification Report

## Dataset
The dataset contains preprocessed tweets about COVID-19.

test_data.csv includes:

X: cleaned tweet text

y: true label (real or fake)

Test: contains TF-IDF or embedding-based test features in pickle format




