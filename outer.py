import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the trained model from a file
clf = joblib.load('model.joblib')

# Convert the user's input to numerical features using HashingVectorizer
vectorizer = HashingVectorizer(stop_words='english', n_features=2**10)
user_input = "NARENDRA MODI IS THE PRIME MINISTER OF INDIA"
user_input_vector = vectorizer.transform([user_input])

# Predict the label of the user's input using the trained model
prediction = clf.predict(user_input_vector)[0]

# Print the predicted label
if prediction == 1:
    print("The article is real.")
else:
    print("The article is fake.")
