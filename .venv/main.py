import voice
import nlp

# Wake word is currently changed here
wake_word = 'computer'

def main():
    while True:
        if (voice.listening(wake_word) == True):
            command = voice.get_command()
            voice.response(command)
            nlp.tagging(command)


if __name__ == "__main__":
    main()