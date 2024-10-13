from xml.etree.ElementInclude import include

from voice import response, listening, get_command
from gpt import ai
from command_classification import comclass

# Wake word is currently changed here
wake_word = 'computer'


# For testing and avoiding speech to text
def text_only():
    while True:
        command = input("User Input: ")
        # Is the input an automation command or a conversational input
        command_class = comclass(command)[0]
        # If conversation, send to AI chat module
        if command_class == "CONVERSATION":
            ai(command)
        # If automation, execute automation command
        elif command_class == "AUTOMATION":
            print("Executing command : " + command)
        # Else return error
        else:
            print("Error")

        # Exit command for testing
        if command == "exit":
            return 0


# Main function that calls the speach to text module
def main():
    while True:
        if (listening(wake_word) == True):
            command = get_command()
            # Is the input an automation command or a conversational input
            command_class = comclass(command)[0]
            # If conversation, send to AI chat module
            if command_class == "CONVERSATION":
                ai(command)
            # If automation, execute automation command
            elif command_class == "AUTOMATION":
                print("Executing command : " + command)
            # Else return error
            else:
                print("Error")



if __name__ == "__main__":
    #main()
    text_only()