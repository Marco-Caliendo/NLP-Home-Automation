from xml.etree.ElementInclude import include

from voice import response, listening, get_command
#from nlp import tagging
from gpt import ai

# Wake word is currently changed here
wake_word = 'computer'


# For debugging and avoiding speech to text
def text_only():
    while True:
        command = input("User Input: ")
        ai(command)

def main():
    while True:
        if (listening(wake_word) == True):
            command = get_command()
            ai(command)
            #response(command)
            #nlp.tagging(command)


if __name__ == "__main__":
    #main()
    text_only()