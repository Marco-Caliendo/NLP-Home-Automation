#%% Import necessary libraries
import pandas as pd
import numpy as np
import nltk
from collections import Counter
from nltk.tokenize import word_tokenize
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import mutual_info_score

#%% Load the dataset from the CSV file
df = pd.read_csv('entities_intents_dataset.csv')

# Display the first few rows of the dataframe
print(df.head())

# Preprocess and Tokenize
nltk.download('punkt')
df['tokens'] = df['sentence'].apply(word_tokenize)

# Create vocabulary and TF-IDF (term frequency-inverse document frequency)
def create_vocab_and_tfidf(df):
    vocab = Counter()
    for tokens in df['tokens']:
        vocab.update(tokens)

    vocab_size = len(vocab)
    word_to_idx = {word: idx + 1 for idx, (word, _) in enumerate(vocab.items())}  # Index starts from 1
    word_to_idx['<PAD>'] = 0  # Padding token

    # TF-IDF calculation
    tfidf_matrix = np.zeros((len(df), vocab_size + 1))
    for i, tokens in enumerate(df['tokens']):
        token_count = Counter(tokens)
        total_tokens = sum(token_count.values())
        for token, count in token_count.items():
            tf = count / total_tokens
            idf = np.log(len(df) / (1 + sum(1 for t in df['tokens'] if token in t)))
            tfidf_matrix[i, word_to_idx[token]] = tf * idf

    return tfidf_matrix, word_to_idx, vocab_size

tfidf_matrix, word_to_idx, vocab_size = create_vocab_and_tfidf(df)

# Encode labels
label_to_idx = {label: idx for idx, label in enumerate(df['intent'].unique())}
y = df['intent'].map(label_to_idx).values

#%% Define Dataset Class
class TextDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input': torch.tensor(self.features[idx], dtype=torch.float),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Create Dataset
dataset = TextDataset(tfidf_matrix, y)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

#%% Define Naive Bayes Class
class NaiveBayes:
    def fit(self, X, y):
        self.class_probs = Counter(y)
        self.word_probs = {}

        for label in self.class_probs.keys():
            X_label = X[y == label]
            total_words = np.sum(X_label, axis=0)
            self.word_probs[label] = (total_words + 1) / (np.sum(total_words) + len(self.class_probs))

    def predict(self, X):
        predictions = []
        for x in X:
            class_scores = {}
            for label in self.class_probs.keys():
                class_scores[label] = np.log(self.class_probs[label] / np.sum(list(self.class_probs.values()))) + np.sum(
                    x * np.log(self.word_probs[label]))
            predictions.append(max(class_scores, key=class_scores.get))
        return np.array(predictions)

# Train and evaluate Naive Bayes
nb_model = NaiveBayes()
nb_model.fit(tfidf_matrix, y)
nb_predictions = nb_model.predict(tfidf_matrix[test_dataset.indices])

#%% Define SVM Class
class SVM(torch.nn.Module):
    def __init__(self, input_size, num_classes):
        super(SVM, self).__init__()
        self.linear = torch.nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

# Train SVM
svm_model = SVM(vocab_size + 1, len(label_to_idx))
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(svm_model.parameters(), lr=0.01)

def train_model(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            inputs = batch['input']
            labels = batch['label']
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

train_model(svm_model, train_loader, criterion, optimizer)

# Evaluate SVM
svm_model.eval()
with torch.no_grad():
    svm_predictions = []
    for batch in test_loader:
        inputs = batch['input']
        outputs = svm_model(inputs)
        _, predicted = torch.max(outputs, 1)
        svm_predictions.extend(predicted.numpy())

# Convert predictions to original labels
svm_predictions = [list(label_to_idx.keys())[pred] for pred in svm_predictions]

#%% Define Decision Tree Class
class DecisionTree:
    def fit(self, X, y):
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y):
        # Base case: if all labels are the same, return that label
        if len(set(y)) == 1:
            return y[0]

        # Base case: if there's no more features or labels are too small
        if X.shape[1] == 0 or len(y) < 2:
            return Counter(y).most_common(1)[0][0]

        best_feature = self._best_feature(X, y)
        tree = {best_feature: {}}

        # For each unique value of the best feature, create a subtree
        for value in set(X[:, best_feature]):
            sub_X = X[X[:, best_feature] == value]
            sub_y = y[X[:, best_feature] == value]
            # Only build the subtree if there are samples for this feature value
            if len(sub_y) > 0:
                tree[best_feature][value] = self._build_tree(sub_X, sub_y)
            else:
                tree[best_feature][value] = Counter(y).most_common(1)[0][0]  # Default to most common class

        return tree

    def _best_feature(self, X, y):
        # Calculate the mutual information for each feature
        num_features = X.shape[1]
        best_feature = None
        best_score = -1

        for feature in range(num_features):
            score = mutual_info_score(X[:, feature], y)  # Using mutual information as a criterion
            if score > best_score:
                best_score = score
                best_feature = feature

        return best_feature

    def predict(self, X):
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        node = self.tree
        while isinstance(node, dict):
            feature = next(iter(node))
            node = node[feature].get(x[feature], Counter(y).most_common(1)[0][0])  # Default to most common class
        return node

# Train Decision Tree
dt_model = DecisionTree()
dt_model.fit(tfidf_matrix, y)

# Evaluate Decision Tree
dt_predictions = dt_model.predict(tfidf_matrix[test_dataset.indices])

#%% Evaluate Naive Bayes
print("Naive Bayes Classification Report:\n", classification_report(y[test_dataset.indices], nb_predictions))

# Evaluate SVM
# Ensure svm_predictions are converted back to string format if they are numeric
if isinstance(svm_predictions[0], int):  # Check if the predictions are numeric
    svm_predictions = [list(label_to_idx.keys())[pred] for pred in svm_predictions]

# Convert y_test to string format as well
y_test = [list(label_to_idx.keys())[label] for label in y[test_dataset.indices]]

# Now evaluate SVM with zero_division parameter
print("SVM Classification Report:\n", classification_report(y_test, svm_predictions, zero_division=0))

# Evaluate Decision Tree
print("Decision Tree Classification Report:\n", classification_report(y[test_dataset.indices], dt_predictions))

#%%
def predict_sentences(sentences, models, word_to_idx, label_to_idx):
    # Preprocess the sentences
    tokenized_sentences = [word_tokenize(sentence) for sentence in sentences]

    # Create TF-IDF matrix for the new sentences
    vocab_size = len(word_to_idx)
    tfidf_matrix = np.zeros((len(sentences), vocab_size))  # Notice we use vocab_size instead of vocab_size + 1

    for i, tokens in enumerate(tokenized_sentences):
        token_count = Counter(tokens)
        total_tokens = sum(token_count.values())
        for token, count in token_count.items():
            if token in word_to_idx:  # Check if the token is in the vocabulary
                tf = count / total_tokens
                idf = np.log(len(df) / (1 + sum(1 for t in df['tokens'] if token in t)))
                tfidf_matrix[i, word_to_idx[token] - 1] = tf * idf  # Adjust index to be zero-based

    # Check the shape of the TF-IDF matrix
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Ensure the dimensions are correct for Naive Bayes
    if tfidf_matrix.shape[1] != vocab_size:
        raise ValueError(f"Expected {vocab_size} features, got {tfidf_matrix.shape[1]}")

    # Predictions from each model
    nb_predictions = models['naive_bayes'].predict(tfidf_matrix)

    # SVM predictions
    svm_model = models['svm']
    svm_inputs = torch.tensor(tfidf_matrix, dtype=torch.float)
    svm_model.eval()
    with torch.no_grad():
        svm_outputs = svm_model(svm_inputs)
        _, svm_predictions = torch.max(svm_outputs, 1)
        svm_predictions = svm_predictions.numpy()

    # Decision Tree predictions
    dt_predictions = models['decision_tree'].predict(tfidf_matrix)

    # Convert predictions back to original labels
    nb_results = [list(label_to_idx.keys())[label] for label in nb_predictions]
    svm_results = [list(label_to_idx.keys())[pred] for pred in svm_predictions]
    dt_results = [list(label_to_idx.keys())[label] for label in dt_predictions]

    return nb_results, svm_results, dt_results


# Example usage remains the same
sentences_to_predict = ["Turn on the lights", "What's the temperature?", "Set the thermostat to 22 degrees"]
models = {
    'naive_bayes': nb_model,
    'svm': svm_model,
    'decision_tree': dt_model
}

try:
    nb_results, svm_results, dt_results = predict_sentences(sentences_to_predict, models, word_to_idx, label_to_idx)

    # Display predictions
    for sentence, nb_pred, svm_pred, dt_pred in zip(sentences_to_predict, nb_results, svm_results, dt_results):
        print(f"Sentence: '{sentence}'")
        print(f"Naive Bayes Prediction: {nb_pred}")
        print(f"SVM Prediction: {svm_pred}")
        print(f"Decision Tree Prediction: {dt_pred}")
        print("---")
except ValueError as e:
    print(e)


