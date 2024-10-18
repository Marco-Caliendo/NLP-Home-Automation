#!/usr/bin/env python3

import ppt_control as ppt

# Current path to power point loaded in system
path = r"C:\Users\calie\PycharmProjects\NLP_Home_Automation\.venv\Sprint_1_Fall_2024.pptx"
presentation = ppt.PPT()

# System defined commands. Returns 0 if input is not a system command
def system_commands(input):
    match input:
        case "exit":
            quit()
        case "system command":
            print("Executing system command : " + input)
        case "open presentation":
            presentation.open_presentation(path)
        case "next slide":
            presentation.next_slide()
        case "previous slide":
            presentation.prev_slide()
        case "end presentation":
            presentation.end_presentation()
        case _:
            return 0
