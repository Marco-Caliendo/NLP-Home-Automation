from voice import listening, get_command
from gpt import ai
from command_classification import comclass
import keyboard_output as key


# Wake word is currently changed here
wake_word = 'computer'


# System defined commands. Returns 0 if input is not a system command
def system_commands(input):
    match input:
        case "exit":
            quit()
        case "system command":
            print("Executing system command : " + input)
            return 1
        case "next slide":
            key.next_slide()
        case "previous slide":
            key.previous_slide()
        case _:
            return 0


# For testing and avoiding speech to text
def text_only():
    while True:
        # Get the command
        command = input("User Input: ")

        # Check if input is a system defined command, if it is, skip the rest of the command loop
        if system_commands(command) != 0:
            continue

        # Is the input an automation command or a conversational input
        command_class = comclass(command)[0]

        # If conversation, send to AI chat module, else ,if automation, execute automation command, else return error
        if command_class == "CONVERSATION":
            ai(command)
        elif command_class == "AUTOMATION":
            print("Executing command : " + command)
        else:
            print("Error")


# Main function that calls the speach to text module
def main():
    while True:
        if (listening(wake_word) == True):
            # Get the command
            command = get_command()

            # Check if input is a system defined command, if it is, skip the rest of the command loop
            if system_commands(command) != 0:
                continue

            # Is the input an automation command or a conversational input
            command_class = comclass(command)[0]

            # If conversation, send to AI chat module, else ,if automation, execute automation command, else return error
            if command_class == "CONVERSATION":
                print(command)
                ai(command)
            elif command_class == "AUTOMATION":
                print("Executing command : " + command)
            else:
                print("Error")





if __name__ == "__main__":
    main()
    #text_only()