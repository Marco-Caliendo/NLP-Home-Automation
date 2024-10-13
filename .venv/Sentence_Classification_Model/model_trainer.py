#%%
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import ast
#%%
# Function to load data from the data file
def load_data(filename):
    with open(filename, "r") as file:
        content = file.read().strip()
        data = ast.literal_eval(content)
    return data

# Define the path to the dataset file
dataset = load_data("data.txt")

# Shuffle the data to ensure randomness
random.shuffle(dataset)

# Split the data into 80% training and 20% testing
split_ratio = 0.8
split_index = int(len(dataset) * split_ratio)

# Split into training data and testing data
data = dataset[:split_index]      # 80% training data
test_data = dataset[split_index:] # 20% test data

# Create word_to_ix mapping
word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent.lower().split():  # Ensure lowercase and split by spaces
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

# Print word to index mapping
print(word_to_ix)

# Define VOCAB_SIZE and NUM_LABELS
VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = len(set(label for _, label in dataset))  # Get unique labels

print(f"VOCAB_SIZE: {VOCAB_SIZE}")
print(f"Training data size: {len(data)}")
print(f"Test data size: {len(test_data)}")
#%%
class Classifier(nn.Module):
    def __init__(self, num_labels, vocab_size, hidden_size):
        super(Classifier, self).__init__()
        self.hidden = nn.Linear(vocab_size, hidden_size)  # Hidden layer
        self.output = nn.Linear(hidden_size, num_labels)  # Output layer

    def forward(self, vec):
        hidden_activation = F.relu(self.hidden(vec))  # Apply ReLU activation
        return F.log_softmax(self.output(hidden_activation), dim=1)
#%%
def make_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence.lower().split():  # Ensure lowercase and split by spaces
        if word in word_to_ix:  # Check if the word exists in the mapping
            vec[word_to_ix[word]] += 1
        else:
            print(f"Warning: '{word}' not found in word_to_ix.")  # Print warning for missing words
    return vec.view(1, -1)
#%%
def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])
#%%
# Define hidden size
hidden_size = 128  # Example size, feel free to change

# Initialize the model with hidden layers
model = Classifier(NUM_LABELS, VOCAB_SIZE, hidden_size)

# Print model parameters
for param in model.parameters():
    print(param)
#%%
# To run the model, pass in a BoW vector
sample = data[0]
vector = make_vector(sample[0], word_to_ix)
log_probs = model(autograd.Variable(vector))
print(log_probs)
#%%
label_to_ix = {"AUTOMATION": 0, "CONVERSATION": 1}
#%%
# Test data before training
for instance, label in test_data:
    vec = autograd.Variable(make_vector(instance, word_to_ix))
    log_probs = model(vec)
    print(log_probs)
#%%
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
#%%
# Training loop
for epoch in range(100):
    for instance, label in data:
        model.zero_grad()
        vec = autograd.Variable(make_vector(instance, word_to_ix))
        target = autograd.Variable(make_target(label, label_to_ix))
        log_probs = model(vec)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()
#%%
# Initialize counters for correct predictions and total predictions
correct_predictions = 0
total_predictions = 0
#%%
for instance, label in test_data:
    vec = autograd.Variable(make_vector(instance, word_to_ix))
    log_probs = model(vec)

    # Get predicted label index
    predicted_label_idx = torch.argmax(log_probs, dim=1).item()
    predicted_label = list(label_to_ix.keys())[list(label_to_ix.values()).index(predicted_label_idx)]

    # Count correct predictions
    total_predictions += 1
    if predicted_label == label:
        correct_predictions += 1

    print(f"Sentence: '{instance}' -> Predicted: {predicted_label}")

# Calculate accuracy
accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
print(f"Accuracy: {accuracy:.2f}")
#%%
print(model)
#%%
export_model_script = torch.jit.script(model)
export_model_script.save('command_classifier.pt')
# Save the word_to_ix mapping to a file
with open("comclass_vocab.txt", "w") as f:
    f.write(str(word_to_ix))
