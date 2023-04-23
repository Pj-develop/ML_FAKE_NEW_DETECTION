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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Load data
data_fake = pd.read_csv("Fake.csv")
data_true = pd.read_csv("True.csv")

# Add label to data
data_fake["class"] = 0
data_true['class'] = 1

# Remove last 10 rows from each dataset for manual testing
data_fake_manual_testing = data_fake.tail(10)
for i in range(23480, 23470, -1):
    data_fake.drop([i], axis=0, inplace=True)

data_true_manual_testing = data_true.tail(10)
for i in range(21416, 21406, -1):
    data_true.drop([i], axis=0, inplace=True)

# Merge data
data_merge = pd.concat([data_fake, data_true], axis=0)
data_merge = data_merge.sample(frac=1).reset_index(drop=True)

# Clean data
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W", " ", text)
    text = re.sub('https?://\S+|www.\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

data_merge['text'] = data_merge['text'].apply(clean_text)

# Split data into train and test sets
x = data_merge['text']
y = data_merge['class']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Vectorize data
vectorizer = TfidfVectorizer()
xv_train = vectorizer.fit_transform(x_train)
xv_test = vectorizer.transform(x_test)
joblib.dump(vectorizer, 'vectorizer.pkl')

# Train models
models = {'Logistic Regression': LogisticRegression(), 'Decision Tree': DecisionTreeClassifier(),
          'Gradient Boosting': GradientBoostingClassifier(), 'Random Forest': RandomForestClassifier()}
best_model = None
best_score = 0

for name, model in models.items():
    model.fit(xv_train, y_train)
    joblib.dump(model, f'{name}.pkl')
    score = model.score(xv_test, y_test)
    if score > best_score:
        best_score = score
        best_model = model
    print(f"{name} score: {score}")
    pred = model.predict(xv_test)
    print(classification_report(y_test, pred))

# Use best model for manual testing
def out_label(n):
    if n == 0: 
        return "FAKE NEWS"
    else : 
        return "IT IS NOT"
