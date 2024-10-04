import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import re
from testing.dataset import train_sentences, train_intents, vocab, intent_map  # Import from dataset.py

# Check if a GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Check dataset consistency
if len(train_sentences) != len(train_intents):
    raise ValueError(f"Number of sentences ({len(train_sentences)}) and intents ({len(train_intents)}) do not match.")

# Check all intents in train_intents are in intent_map
unique_intents = set(train_intents)
mapped_intents = set(intent_map.values())
if not unique_intents.issubset(mapped_intents):
    missing = unique_intents - mapped_intents
    raise ValueError(f"Train intents contain values {missing} not present in intent_map.")

# Intent Classification Model
class IntentClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_intents):
        super(IntentClassifier, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=2, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_intents)

    def forward(self, input_seq, lengths):
        embedded = self.embedding(input_seq)
        packed_input = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed_input)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        intent_logits = self.fc(lstm_output[:, -1, :])  # Use the last output
        return intent_logits

# Tokenizer function
def tokenize(sentence):
    # Return the index of the word, or <UNK> if the word is not in the vocabulary
    return [vocab.get(word, vocab['<UNK>']) for word in sentence.lower().split()]

# Custom Dataset class to handle variable length inputs
class CustomDataset(Dataset):
    def __init__(self, sentences, intents):
        self.sentences = [torch.tensor(tokenize(sent), dtype=torch.long) for sent in sentences]
        self.intents = torch.tensor(intents, dtype=torch.long)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.intents[idx]

# Create dataset instance
dataset = CustomDataset(train_sentences, train_intents)

# Collate function to pad sequences dynamically in each batch
def collate_fn(batch):
    sentences, intents = zip(*batch)
    sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=vocab['<PAD>'])
    lengths = torch.tensor([len(s[s != vocab['<PAD>']]) for s in sentences], dtype=torch.long)
    intents = torch.stack(intents)
    return sentences_padded, lengths, intents

# DataLoader with custom collate function
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

# Slot Extraction Function
def extract_slots(sentence):
    sentence = sentence.lower()
    device_pattern = r"\blight|thermostat|fan|air conditioner|door\b"
    location_pattern = r"\bkitchen|bedroom|living room|office\b"
    device = re.search(device_pattern, sentence)
    location = re.search(location_pattern, sentence)
    return {'device': device.group(0) if device else None, 'location': location.group(0) if location else None}

# Simulate controlling devices based on intent
def control_device(intent, slots):
    device = slots['device']
    location = slots['location']
    if intent == 'TurnOnDevice':
        return f"Turning on {device} in the {location}" if device and location else "Command unclear."
    elif intent == 'TurnOffDevice':
        return f"Turning off {device} in the {location}" if device and location else "Command unclear."
    elif intent == 'QueryDeviceState':
        return f"The {device} in the {location} is currently off"  # Mock response
    else:
        return "Unknown intent"


# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token  # Set pad token
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
gpt2_model.eval()


# Generate GPT-2 response for fallback
def generate_gpt_response(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt').to(device)
    # Create attention mask
    attention_mask = (inputs != tokenizer.pad_token_id).long()

    outputs = gpt2_model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=50,
        do_sample=True,
        temperature=0.7
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Model hyperparameters
input_size = len(vocab)
hidden_size = 128
num_intents = len(intent_map)
learning_rate = 0.001
epochs = 100

# Initialize model, loss function, and optimizer
intent_model = IntentClassifier(input_size, hidden_size, num_intents).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(intent_model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    total_loss = 0
    for input_batch, lengths, target_batch in data_loader:
        optimizer.zero_grad()
        input_batch, target_batch = input_batch.to(device), target_batch.to(device)
        output = intent_model(input_batch, lengths)
        loss = criterion(output, target_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(data_loader)
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}')

# Save model function
def save_model(model, path):
    torch.save(model.state_dict(), path)

# Load model function
def load_model(model, path):
    model.load_state_dict(torch.load(path))
    model.to(device)
    model.eval()

# Save the model after training
#save_model(intent_model, 'intent_classifier.pth')

# Function to process user input
def process_user_input(user_input):
    # Tokenize the input
    input_tensor = torch.tensor(tokenize(user_input), dtype=torch.long).unsqueeze(0).to(device)
    lengths = [len(input_tensor[input_tensor != vocab['<PAD>']])]

    # Classify intent
    intent_logits = intent_model(input_tensor, lengths)
    intent_index = torch.argmax(intent_logits, dim=1).item()
    intent = list(intent_map.keys())[intent_index]

    # Extract slots (device, location)
    slots = extract_slots(user_input)

    # Execute the corresponding action or provide fallback response
    if intent in ['TurnOnDevice', 'TurnOffDevice', 'QueryDeviceState']:
        response = control_device(intent, slots)
    else:
        response = generate_gpt_response(user_input)  # Use GPT-2 for fallback
    return response

# Example interaction
print("Home Automation Bot is ready! Type 'exit' to quit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break
    response = process_user_input(user_input)
    print("Bot:", response)
