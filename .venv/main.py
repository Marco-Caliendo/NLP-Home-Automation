#from xml.etree.ElementInclude import include

from voice import response, listening, get_command
from gpt import ai
from command_classification import comclass

# Wake word is currently changed here
wake_word = 'computer'


# System defined commands. Returns 0 if input is not a system command
def system_commands(input):
    match input:
        case "exit":
            sys.exit()
        case _:
            return 0


# For testing and avoiding speech to text
def text_only():
    while True:
        # Get the command
        command = input("User Input: ")

        # Check if input is a system defined command
        if system_commands(command) == 0:
            # Is the input an automation command or a conversational input
            command_class = comclass(command)[0]
            # If conversation, send to AI chat module, else ,if automation, execute automation command, else return error
            if command_class == "CONVERSATION":
                ai(command)
            elif command_class == "AUTOMATION":
                print("Executing command : " + command)
            else:
                print("Error")
        else:
            print("Executing System Command : " + command)


# Main function that calls the speach to text module
def main():
    while True:
        if (listening(wake_word) == True):
            # Get the command
            command = get_command()

            # Check if input is a system defined command
            if system_commands(command) == 0:
                # Is the input an automation command or a conversational input
                command_class = comclass(command)[0]
                # If conversation, send to AI chat module, else ,if automation, execute automation command, else return error
                if command_class == "CONVERSATION":
                    ai(command)
                elif command_class == "AUTOMATION":
                    print("Executing command : " + command)
                else:
                    print("Error")
        else:
            print("Executing System Command : " + command)



if __name__ == "__main__":
    #main()
    text_only()