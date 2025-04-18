#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from math import log10
import re
from emoji import UNICODE_EMOJI
import emoji
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# In[27]:


df = pd.read_excel("data.xlsx")
print(df.head(10))

df_test= pd.read_csv("test.csv")
# #Preprocess
# 

# In[28]:


def super_simple_preprocess(text):
  # lowercase
  text = text.lower()
  # remove non alphanumeric characters
  text = re.sub('[^A-Za-z0-9 ]+',' ', text)
  return text
def process_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(text)
    lemmatized_tokens = [lemmatizer.lemmatize(word)
                         for word in tokens
                         if word.lower() not in stop_words]

    return ' '.join(lemmatized_tokens)


# ## Handling #Tag

# In[29]:


def memo(f):
    "Memoize function f."
    table = {}
    def fmemo(*args):
        if args not in table:
            table[args] = f(*args)
        return table[args]
    fmemo.memo = table
    return fmemo

def test(verbose=None):
    """Run some tests, taken from the chapter.
    Since the hillclimbing algorithm is randomized, some tests may fail."""
    import doctest
    doctest.testfile('ngrams-test.txt', verbose=verbose)
@memo
def segment(text):
    "Return a list of words that is the best segmentation of text."
    text = text.lower()
    if not text:
      return []
    candidates = ([first]+segment(rem) for first,rem in splits(text))
    return max(candidates, key=Pwords)

def splits(text, L=20):
    "Return a list of all possible (first, rem) pairs, len(first)<=L."
    return [(text[:i+1], text[i+1:])
            for i in range(min(len(text), L))]

def final_output(text):
    seg = segment(str(text)[1:])
    if(len(seg)>0):
      return " ".join(seg)
    return " "

def Pwords(words):
    "The Naive Bayes probability of a sequence of words."
    return product(Pw(w) for w in words)

#### Support functions (p. 224)

def product(nums):
    "Return the product of a sequence of numbers."
    ans=0
    for i in nums:
      ans+=log10(i)
    return ans

class Pdist(dict):
    "A probability distribution estimated from counts in datafile."
    def __init__(self, data=[], N=None, missingfn=None):
        for d in data:
            if(len(d)!=2):
              continue
            self[d[0]] = self.get(d[0], 0) + int(d[1])
        self.N = float(N or sum(self.itervalues()))
        self.missingfn = missingfn or (lambda k, N: 1./N)
    def __call__(self, key):
        if key in self: return self[key]/self.N
        else: return self.missingfn(key, self.N)

def datafile(name, sep='\t'):
    "Read key,value pairs from file."
    final = []
    f = open(name,'r')
    while True:
      line = f.readline()
      if not line:
        break
      final.append(line.split('\t'))
    return final

def avoid_long_words(key, N):
    "Estimate the probability of an unknown word."
    return 10./(N * 10**len(key))

N = 1024908267229 ## Number of tokens
Pw  = Pdist(datafile('count_1w.txt'), N, avoid_long_words)


# ## Handling URL & emoji

# In[30]:


def remove_urls(text, replacement_text=""):
    # Define a regex pattern to match URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # Use the sub() method to replace URLs with the specified replacement text
    text_without_urls = url_pattern.sub(replacement_text, text)
    text_without_urls_emoji = emoji.demojize(text_without_urls)
    return text_without_urls_emoji
print(remove_urls("i am www.useless.com Let's grab some lunch: 🍕 or 🍜 or 🍱?"))


# In[31]:


def my_preprocessor(text):
  text = remove_urls(text)
  text = re.sub(r'#[A-Za-z0-9]+', lambda m: final_output(m.group()), text)
  text = super_simple_preprocess(text)
  # text = process_text(text)
  return text


# In[32]:


df['preprocessed_tweet'] = df['tweet'].apply(my_preprocessor)
print(df[5:10])

df_test['preprocessed_tweet'] = df_test['tweet'].apply(my_preprocessor)
# In[33]:


print(df["tweet"][3])
print(df["preprocessed_tweet"][3])
print(final_output("#IndiaFightsCorona"))


# In[34]:


X = df['preprocessed_tweet']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=49)

X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=49)

X_test = df_test['preprocessed_tweet']
# y_test = df_test['label']
# In[35]:


# Create DataFrames for train, test, and validation sets
train_df = pd.DataFrame({'X': X_train, 'y': y_train})
test_df = pd.DataFrame({'X': X_test})
val_df = pd.DataFrame({'X': X_val, 'y': y_val})
# Export DataFrames to CSV files
train_df.to_csv('train_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)
val_df.to_csv('validate_data.csv', index=False)


# In[36]:


count_vectorizer = CountVectorizer()
# Fit and transform the 'text' column to get the tokens
tokens = count_vectorizer.fit_transform(pd.concat([train_df, val_df], ignore_index=True)['X'])
# Get the feature names (vocabulary)
vocabulary = count_vectorizer.get_feature_names_out()


# In[37]:


#Train

tfidf_vectorizer = TfidfVectorizer(vocabulary=vocabulary)
tfidf_matrix = tfidf_vectorizer.fit_transform(X_train)

# Convert the TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Display the DataFrame with TF-IDF vector representations
dbfile = open('Train', 'ab')
pickle.dump(tfidf_df, dbfile)
dbfile.close()
#Validate
# Create TF-IDF vectorizer with custom vocabulary

tfidf_matrix = tfidf_vectorizer.fit_transform(X_val)

# Convert the TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Display the DataFrame with TF-IDF vector representations
dbfile = open('Validate', 'ab')
pickle.dump(tfidf_df, dbfile)
dbfile.close()

tfidf_matrix = tfidf_vectorizer.fit_transform(X_test)

# Convert the TF-IDF matrix to a DataFrame
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
print(tfidf_df)
# Display the DataFrame with TF-IDF vector representations
dbfile = open('Test', 'ab')
pickle.dump(tfidf_df, dbfile)
dbfile.close()


# In[ ]:




