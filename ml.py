import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import re 
import string
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer



data_fake=pd.read_csv("Fake.csv")
data_true=pd.read_csv("True.csv")

data_fake["class"]=0
data_true['class']=1

data_merge=pd.concat([data_fake,data_true],axis=0)

def clean_text(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub("\\W"," ",text)
    text=re.sub('https?://\S+|www.\.\S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]'% re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    return text

data_merge['text'] = data_merge['text'].apply(clean_text)

x=data_merge['text']
y=data_merge['class']

x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)

pipeline = Pipeline([
    ('vectorization', TfidfVectorizer()),
    ('classifier', RandomForestClassifier())
])

parameters = {
    'vectorization__max_df': (0.25, 0.5, 0.75),
    'vectorization__ngram_range': ((1, 1), (1, 2)),
    'classifier__C': (0.1, 1, 10)
}

grid_search = GridSearchCV(pipeline, parameters, cv=5)
grid_search.fit(x_train, y_train)

print(f"Best score: {grid_search.best_score_}")
print(f"Best parameters: {grid_search.best_params_}")

best_model = grid_search.best_estimator_
joblib.dump(best_model, 'best_model.pkl')

def out_label(n):
    if n==0: 
        return "FAKE NEWS"
    else : 
        return "IT IS NOT A REAL NEWS"

def manual_testing(news):
    model = joblib.load('best_model.pkl')
    news_cleaned = clean_text(news)
    prediction = model.predict([news_cleaned])[0]
    label = out_label(prediction)
    print(f"\n\nPrediction: {label}")

news="Angolan president arrives in Japan"
manual_testing(news)
