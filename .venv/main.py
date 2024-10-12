from xml.etree.ElementInclude import include

from voice import response, listening, get_command
#from nlp import tagging
from gpt import ai
from command_classification import comclass

# Wake word is currently changed here
wake_word = 'computer'


# For debugging and avoiding speech to text
def text_only():
    while True:
        command = input("User Input: ")
        command_class = comclass(command)[0]
        if command_class == "CONVERSATION":
            ai(command)
        elif command_class == "AUTOMATION":
            print("Executing command : " + command)
        else:
            print("Error")

        if command == "exit":
            return 0


def main():
    while True:
        if (listening(wake_word) == True):
            command = get_command()
            command_class = comclass(command)[0]
            if command_class == "CONVERSATION":
                ai(command)
            elif command_class == "AUTOMATION":
                print("Executing command : " + command)
            else:
                print("Error")
            #response(command)
            #nlp.tagging(command)



if __name__ == "__main__":
    #main()
    text_only()