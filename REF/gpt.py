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


def wordopt(text):
    text=text.lower()
    text=re.sub('\[.*?\]','',text)
    text=re.sub("\\W"," ",text)
    text=re.sub('https?://\S+|www.\.\S+','',text)
    text=re.sub('<.*?>+','',text)
    text=re.sub('[%s]'% re.escape(string.punctuation),'',text)
    text=re.sub('\n','',text)
    text=re.sub('\w*\d\w*','',text)
    return text


from sklearn.feature_extraction.text import TfidfVectorizer
vectorization=TfidfVectorizer()


def out_label(n):
    if n==0: 
        return "FAKE NEWS"
    else : 
        return "IT IS NOT A REAL NEWS"


def manual_testing(news):
    testing_news={"text":[news]}
    new_def_test=pd.DataFrame(testing_news)
    new_def_test['text']=new_def_test['text'].apply(wordopt)

    # load the fitted vectorizer
    vectorization = joblib.load('vectorizer.pkl')
    
    new_x_test=new_def_test["text"]
    new_xv_test=vectorization.transform(new_x_test)
    
    model = joblib.load('mod_4.pkl')
    pred_RF = model.predict(new_xv_test)
    return print("\n\n \n RF: {}\n".format(out_label(pred_RF[0])))


news="Thousands of farmers had camped at Delhi's borders since last November and dozens died from heat, cold and Covid Farmers say the laws will allow the entry of private players in farming and that will hurt their income. Friday's surprise announcement marks a major U-turn as the government had not taken any initiative to talk to farmers in recent months. And Mr Modi's ministers have been steadfastly insisting that the laws were good for farmers and there was no question of taking them back. Farm unions are seeing this as a huge victory. But experts say the upcoming state elections in Punjab and Uttar Pradesh - both have a huge base of farmers - may have forced the decision. The announcement on Friday morning came on a day Sikhs - the dominant community in Punjab - are celebrating the birth anniversary of Guru Nanak, the founder of Sikhism."
manual_testing(news)
