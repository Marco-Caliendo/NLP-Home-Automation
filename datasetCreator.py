import os
import nltk
from nltk.corpus import gutenberg

# Ensure that the NLTK gutenberg corpus is downloaded
nltk.download('gutenberg')

# Create a directory to store the text files
os.makedirs('data', exist_ok=True)

# Iterate through the fileids in the Reuters corpus
for fileid in gutenberg.fileids():
    # Retrieve the text of the news article
    article_text = ' '.join(gutenberg.words(fileid))

    # Construct the filename (you can also use
    # the categories in the filename if needed)
    filename = f'data/{fileid.replace("/", "_")}.txt'

    # Write the article text to a file
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(article_text)

print("If everything went fine, gutenberg articles are saved as text files.")