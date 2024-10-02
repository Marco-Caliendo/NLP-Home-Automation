import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import string

#Key Changes For Faster Training:
#Batching and Padding: prepare_batch prepares input and target batches with padding and corresponding sequence lengths.
#Packed Sequences: The forward method handles packed sequences to avoid unnecessary computation on padded tokens.
#Gradient Clipping: Applied gradient clipping to avoid exploding gradients.
#Mixed Precision Training: Using torch.cuda.amp with automatic casting and gradient scaling to accelerate training on GPUs with mixed precision.
#Reduced Hidden Size: Reduced the hidden size to 64 for faster training.
#Ignored Padding in Loss: ignore_index=vocab['<PAD>'] ensures padding tokens are ignored during loss calculation

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Chatbot Model Definition
class ChatbotModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(ChatbotModel, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        # Embedding layer
        self.embedding = nn.Embedding(input_size, hidden_size)

        # LSTM layer
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, batch_first=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input_seq, hidden_state, lengths):
        embedded = self.embedding(input_seq)
        packed_embedded = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        packed_output, hidden_state = self.lstm(packed_embedded, hidden_state)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        output = self.fc(output)
        return output, hidden_state

    def init_hidden(self, batch_size):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size).to(device))

#######################################################################################################################

# Tokenize and preprocess text data
def tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text.split()

# Build a vocabulary from the dataset
def build_vocab(dataset):
    all_words = []
    for sentence in dataset:
        all_words.extend(tokenize(sentence))

    vocab = {word: i for i, word in enumerate(set(all_words))}
    vocab['<PAD>'] = len(vocab)  # Padding token
    vocab['<SOS>'] = len(vocab)  # Start of sentence
    vocab['<EOS>'] = len(vocab)  # End of sentence
    return vocab

# Prepare batch with padding
def prepare_batch(input_tensors, target_tensors, batch_size):
    idx = np.random.choice(len(input_tensors), batch_size, replace=False)
    batch_inputs = [input_tensors[i] for i in idx]
    batch_targets = [target_tensors[i] for i in idx]

    # Pad the input and target sequences to the same length
    padded_inputs = pad_sequence(batch_inputs, batch_first=True, padding_value=vocab['<PAD>'])
    padded_targets = pad_sequence(batch_targets, batch_first=True, padding_value=vocab['<PAD>'])
    lengths = [len(tensor) for tensor in batch_inputs]

    return padded_inputs, padded_targets, lengths

# Training function
def train(chatbot_model, input_tensor, target_tensor, optimizer, criterion, batch_size, lengths, scaler):
    hidden_state = chatbot_model.init_hidden(batch_size)

    optimizer.zero_grad()

    input_tensor = input_tensor.to(device)
    target_tensor = target_tensor.to(device)

    with autocast():
        output, hidden_state = chatbot_model(input_tensor, hidden_state, lengths)

        output = output.view(-1, output.size(-1))  # (batch_size * seq_len, vocab_size)
        target_tensor = target_tensor.view(-1)     # (batch_size * seq_len)

        loss = criterion(output, target_tensor)

    scaler.scale(loss).backward()

    # Gradient clipping to avoid exploding gradients
    torch.nn.utils.clip_grad_norm_(chatbot_model.parameters(), max_norm=2.0)

    scaler.step(optimizer)
    scaler.update()

    return loss.item()

# Generate response function
def generate_response(chatbot_model, input_seq, vocab, max_len=20):
    chatbot_model.eval()

    input_tensor = torch.tensor([vocab[word] for word in tokenize(input_seq)], dtype=torch.long).unsqueeze(0).to(device)
    hidden_state = chatbot_model.init_hidden(1)  # Batch size of 1 for inference

    response = []
    for _ in range(max_len):
        output, hidden_state = chatbot_model(input_tensor, hidden_state, [len(input_tensor[0])])

        _, top_word = torch.max(output[:, -1, :], dim=1)

        word = list(vocab.keys())[list(vocab.values()).index(top_word.item())]
        response.append(word)

        if word == '<EOS>':
            break

        input_tensor = torch.tensor([[top_word.item()]], dtype=torch.long).to(device)

    return ' '.join(response[:-1])  # Remove <EOS> from the response

#######################################################################################################################

# Example dataset
with open('iodata.csv', 'r', encoding='utf-8') as f:
    dataset = f.readlines()
print(dataset[:10])

# Build vocab and tokenize the data
vocab = build_vocab(dataset)

# Convert dataset to tensors
input_tensors = []
target_tensors = []
for sentence in dataset:
    tokenized = tokenize(sentence)
    input_tensors.append(torch.tensor([vocab['<SOS>']] + [vocab[word] for word in tokenized], dtype=torch.long))
    target_tensors.append(torch.tensor([vocab[word] for word in tokenized] + [vocab['<EOS>']], dtype=torch.long))

# Define hyperparameters
input_size = len(vocab)
hidden_size = 128
output_size = len(vocab)
n_layers = 2
batch_size = 8
learning_rate = 0.01
epochs = 100

# Initialize model, loss function, and optimizer
chatbot_model = ChatbotModel(input_size, hidden_size, output_size, n_layers).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<PAD>']).to(device)
optimizer = optim.Adam(chatbot_model.parameters(), lr=learning_rate)
scaler = GradScaler()

# Training loop
for epoch in range(epochs):
    total_loss = 0
    chatbot_model.train()

    for _ in range(len(input_tensors) // batch_size):
        batch_input, batch_target, lengths = prepare_batch(input_tensors, target_tensors, batch_size)
        loss = train(chatbot_model, batch_input, batch_target, optimizer, criterion, batch_size, lengths, scaler)
        total_loss += loss

    print(f'Epoch {epoch}, Loss: {total_loss / (len(input_tensors) // batch_size)}')

# Generate a response
user_input = "turn off the lights"
response = generate_response(chatbot_model, user_input, vocab)
print("Bot:", response)
