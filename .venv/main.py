#!/usr/bin/env python3

from voice import listening, get_command
from command_classification import comclass
import system_commands as sc
import home_assistant_module as ha
import llm


# Wake word is currently changed here
wake_word = 'computer'


# For testing and avoiding speech to text
def text_only():
    while True:
        # Get the command
        command = input("User Input: ")

        # Check if input is a system defined command, if it is, skip the rest of the command loop
        if sc.system_commands(command) != 0:
            continue

        # Is the input an automation command or a conversational input
        command_class = comclass(command)[0]

        # If conversation, send to AI chat module, else ,if automation, execute automation command, else return error
        if command_class == "CONVERSATION":
            llm.red_ai(command)
        elif command_class == "AUTOMATION":
            ha.execute_command(command)
        else:
            print("Error")


# Main function that calls the speach to text module
def main():
    while True:
        if listening(wake_word) == True:
            # Get the command
            command = get_command()

            # Check if input is a system defined command, if it is, skip the rest of the command loop
            if sc.system_commands(command) != 0:
                continue

            # Is the input an automation command or a conversational input
            command_class = comclass(command)[0]

            # If conversation, send to AI chat module, else ,if automation, execute automation command, else return error
            if command_class == "CONVERSATION":
                print(command)
                llm.red_ai(command)
            elif command_class == "AUTOMATION":
                ha.execute_command(command)
            else:
                print("Error")





if __name__ == "__main__":
    #main()
    text_only()
