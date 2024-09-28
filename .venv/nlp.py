import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# This function tags each word in the command for further proccessing
def tagging(command):
    data = nltk.sent_tokenize(command)
    data = [nltk.word_tokenize(word) for word in data]
    data = [nltk.pos_tag(word) for word in data]
    print("Tags: ", data)

