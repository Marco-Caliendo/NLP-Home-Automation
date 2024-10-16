import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score  # Import accuracy_score

# Load the dataset from a CSV file
data = pd.read_csv('intents_dataset.csv')

# Split into sentences and labels
sentences = data['sentence'].tolist()
labels = data['intent'].tolist()

# Vectorize the sentences
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sentences)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')  # Output the accuracy

# Function to predict intent
def predict_intent(command):
    command_vec = vectorizer.transform([command])
    return model.predict(command_vec)[0]

# Example prediction
command = "Turn on the living room lights."
predicted_intent = predict_intent(command)
print(predicted_intent)  # Output: 'turn_on'
