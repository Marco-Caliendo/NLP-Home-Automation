import keyboard


# Simulate a key press (e.g., "a")
#>>keyboard.press('a')
#>>keyboard.release('a')
# Simulate typing a sentence
#>>keyboard.write('Hello World!')
# Simulate pressing combination (e.g., "ctrl+c")
#>>keyboard.press_and_release('ctrl+c')


def next_slide():
    keyboard.press('right')
    keyboard.release('right')

def prev_slide():
    keyboard.press('left')
    keyboard.release('left')


