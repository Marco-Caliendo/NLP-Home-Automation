#!/usr/bin/env python3

import torch
import torch.nn.functional as F
import torch.nn as nn
import ast

# Define the same Classifier class used in training
class Classifier(nn.Module):
    def __init__(self, num_labels, vocab_size, hidden_size):
        super(Classifier, self).__init__()
        self.hidden = nn.Linear(vocab_size, hidden_size)
        self.output = nn.Linear(hidden_size, num_labels)

    def forward(self, vec):
        hidden_activation = F.relu(self.hidden(vec))
        return F.log_softmax(self.output(hidden_activation), dim=1)

# Load the saved model
model = torch.jit.load('command_classifier.pt')
model.eval()  # Set the model to evaluation mode

# Load the vocab and label mappings
def load_vocab(filename):
    with open(filename, "r") as file:
        content = file.read().strip()
        return ast.literal_eval(content)

# Load vocab mappings
comclass_vocab = load_vocab("comclass_vocab.txt")
comclass_labels = {"AUTOMATION": 0, "CONVERSATION": 1}

# Prepare a function to convert input to vector
def make_vector(sentence, comclass_vocab):
    vector = torch.zeros(len(comclass_vocab))
    for word in sentence.lower().split():  # Ensure lowercase and split by spaces
        if word in comclass_vocab:  # Check if the word exists in the mapping
            vector[comclass_vocab[word]] += 1
    return vector.view(1, -1)

# Function to classify a new sentence
def comclass(sentence):
    vector = make_vector(sentence, comclass_vocab)  # Create the vector
    log_probs = model(vector)  # Get log probabilities from the model

    # Get predicted label index
    predicted_label_id = torch.argmax(log_probs, dim=1).item()
    predicted_label = list(comclass_labels.keys())[list(comclass_labels.values()).index(predicted_label_id)]

    return predicted_label, log_probs
