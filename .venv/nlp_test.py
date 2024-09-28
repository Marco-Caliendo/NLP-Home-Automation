import random
import nltk
from nltk import bigrams, FreqDist, ConditionalFreqDist
import os
import string
from itertools import islice
import heapq

nltk.download('punkt')

## Keep your training documents in a folder named 'data'
input_data_dir = "data"

# String of punctuation without the full stop
punctuation = string.punctuation.replace('.', '')  # Retain the full stop

def is_hidden(filepath):
    return os.path.basename(filepath).startswith('.')

text_data=""
for filename in os.listdir(input_data_dir):
    filepath = os.path.join(input_data_dir, filename)
    if not is_hidden(filepath):
        with open(filepath) as infile:
            for line in infile:
                if line.strip():  # Check if line is not just whitespace
                    # Remove all punctuation except full stops
                    for char in punctuation:
                        line = line.replace(char, '')
                    text_data += line

len(text_data)

# Tokenize the text into words
# Lowercasing for consistency
words = nltk.word_tokenize(text_data.lower())

# Generate bigrams
bi_grams = list(bigrams(words))

# Calculate frequency distribution for each bigram
bi_gram_freq_dist = FreqDist(bi_grams)

# Print the first five elements of the dictionary
first_five_items = list(islice(bi_gram_freq_dist.items(), 5))
for item in first_five_items:
    print(item)

# Compute conditional frequency distribution of bigrams
bi_gram_freq = ConditionalFreqDist(bi_grams)

topk=3
# Create a dictionary to hold the top topk bigrams for each first word
top_bigrams_per_first_word = {}

# Iterate over the bigram frequency distribution
for (first_word, second_word), freq in bi_gram_freq_dist.items():
    # Initialize an empty heap for the first_word if it doesn't exist
    if first_word not in top_bigrams_per_first_word:
        top_bigrams_per_first_word[first_word] = []

    # Add to the heap and maintain top topk
    heapq.heappush(top_bigrams_per_first_word[first_word],
                   (freq, second_word))
    if len(top_bigrams_per_first_word[first_word]) > topk:
        heapq.heappop(top_bigrams_per_first_word[first_word])

# Convert the heap to a simple list for each first word
for first_word in top_bigrams_per_first_word:
    sorted_bigrams = sorted(
        top_bigrams_per_first_word[first_word], reverse=True)
    top_bigrams_list = []
    for freq, second_word in sorted_bigrams:
        top_bigrams_list.append(second_word)
    top_bigrams_per_first_word[first_word] = top_bigrams_list

# Use these filtered bigrams to create a ConditionalFreqDist
filtered_bi_grams = []
for first_word in top_bigrams_per_first_word:
    for second_word in top_bigrams_per_first_word[first_word]:
        filtered_bi_grams.append((first_word, second_word))

bi_gram_freq = ConditionalFreqDist(filtered_bi_grams)

def generate_sentence(word, num_words):
    word =word.lower()
    for _ in range(num_words):
        print(word, end=' ')
        next_words = [item for item, freq in bi_gram_freq[word].items()]
        if len(next_words) > 0:
            # Randomly choose a next word
            word = random.choice(next_words)
        else:
            break  # Break if the word has no following words
    print()

generate_sentence('Hello', 10)