#!/usr/bin/env python3

import speech_recognition as sr
import pyaudio
import pyttsx3
import time


'''
This section of the code proccesses audio input and output for the system
Currently using Google Voice Recognition, can be easily changed
'''


# Define r as the voice recognizer
recognize = sr.Recognizer()
# Represents the energy level threshold for sounds. Values below this threshold are considered silence, and values above this threshold are considered speech. Can be changed.
recognize.energy_threshold = 4000
# Enable or disable dynamic energy threshold
recognize.dynamic_energy_threshold = True
# Represents the minimum length of silence (in seconds) that will register as the end of a phrase. Can be changed.
recognize.pause_threshold = 0.5

with sr.Microphone() as source:
    recognize.adjust_for_ambient_noise(source)

# Listen for the wake word
def listening(wake_word):
    recognize.pause_threshold = 1
    try:
        with sr.Microphone() as source:
            print("Listening")
            voice = recognize.listen(source,2,1)
            audio = recognize.recognize_google(voice)
            audio = audio.lower()
            if wake_word in audio:
                return True
            else:
                return False
    except:
        return False


# Get the command after the wake word has been heard
def get_command():
    recognize.pause_threshold = 1
    try:
        with sr.Microphone() as source:
            print("Getting Command")
            voice = recognize.listen(source,10)
            audio = recognize.recognize_google(voice)
            audio = audio.lower()
    except:
        return "Im sorry, I could not understand that"
    return audio


# Respond to the command
def response(command):
    try:
        print("The Computer Heard: " + command)
    except sr.UnknownValueError:
        print("Im sorry I could not understand that")
    except sr.RequestError as e:
        print("Recognition error; {0}".format(e))

