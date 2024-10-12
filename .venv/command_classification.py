#%%
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

# Load the word_to_ix and label_to_ix mappings
def load_vocab(filename):
    with open(filename, "r") as file:
        content = file.read().strip()
        return ast.literal_eval(content)

# Load vocab mappings (assumes you saved it in a separate file, e.g., "vocab.txt")
word_to_ix = load_vocab("word_to_ix.txt")  # You need to save this separately
label_to_ix = {"AUTOMATION": 0, "CONVERSATION": 1}  # Ensure this matches your labels

# Prepare a function to convert input to BoW vector
def make_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence.lower().split():  # Ensure lowercase and split by spaces
        if word in word_to_ix:  # Check if the word exists in the mapping
            vec[word_to_ix[word]] += 1
    return vec.view(1, -1)

# Function to classify a new sentence
def classify_sentence(sentence):
    vector = make_vector(sentence, word_to_ix)  # Create the BoW vector
    log_probs = model(vector)  # Get log probabilities from the model

    # Get predicted label index
    predicted_label_idx = torch.argmax(log_probs, dim=1).item()
    predicted_label = list(label_to_ix.keys())[list(label_to_ix.values()).index(predicted_label_idx)]

    return predicted_label, log_probs

def comclass(input):
    predicted_label = classify_sentence(input)
    return predicted_label

#%%
#print(comclass("close the doors")[0])

