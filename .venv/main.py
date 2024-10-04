from xml.etree.ElementInclude import include

from voice import response, listening, get_command
#from nlp import tagging
from gpt import ai

# Wake word is currently changed here
wake_word = 'computer'

def main():
    while True:
        if (listening(wake_word) == True):
            command = get_command()
    #        response(command)
            ai(command)
    #        nlp.tagging(command)

    #For debugging and avoiding the wake word requirement
    #    command = get_command()
    #    ai(command)


if __name__ == "__main__":
    main()