import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re 
import string
import joblib
from sklearn.ensemble import RandomForestClassifier

# Import the TfidfVectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer

# Create an instance of the TfidfVectorizer object
vectorization = TfidfVectorizer()

# Load the trained random forest classifier from a saved file
clf = joblib.load('mod_4.pkl')

# Define a function to clean and preprocess the text
def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]','',text)
    text = re.sub("\\W"," ",text)
    text = re.sub('https?://\S+|www.\.\S+','',text)
    text = re.sub('<.*?>+','',text)
    text = re.sub('[%s]' % re.escape(string.punctuation),'',text)
    text = re.sub('\n','',text)
    text = re.sub('\w*\d\w*','',text)
    return text

# Define a function to output the label based on the predicted value
def out_label(n):
    if n == 0: 
        return "FAKE NEWS"
    else: 
        return "IT IS NOT A REAL NEWS"

# Define a function to manually test the classifier on a given news article
def manual_testing(news):
    # Create a dictionary with a single key "text" and value "news"
    testing_news = {"text": [news]}
    
    # Convert the dictionary to a Pandas DataFrame
    new_def_test = pd.DataFrame(testing_news)
    
    # Clean and preprocess the text in the DataFrame
    new_def_test['text'] = new_def_test['text'].apply(wordopt)
    
    # Get the cleaned text from the DataFrame
    new_x_test = new_def_test["text"]
    
    # Vectorize the cleaned text using the TfidfVectorizer
    vectorizer = TfidfVectorizer()

# fit the vectorizer on the training data
    new_x_test = vectorizer.fit_transform(new_x_test)

    # transform the test data
    new_xv_test = vectorizer.transform(new_x_test)
    # new_xv_test = vectorization.transform(new_x_test)
    # new_xv_test = vectorization.fit_transform(new_x_test)
    
    # Use the trained classifier to predict the label of the article
    pred_RF = clf.predict(new_xv_test)
    
    # Output the label based on the predicted value
    return print("\n\n RF: {}\n".format(out_label(pred_RF[0])))

# Test the classifier on a news article
news = "US supreme court supports abortion laws"
manual_testing(news)
