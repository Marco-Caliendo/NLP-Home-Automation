#!/usr/bin/env python3

import speech_recognition as sr
import pyaudio
import pyttsx3
import time


'''
This section of the code proccesses audio input and output for the system
Currently using Google Voice Recognition, can be easily changed
'''


# Start the text to speach audio output engine
engine = pyttsx3.init()
""" RATE"""
rate = engine.getProperty('rate')   # getting details of current speaking rate
engine.setProperty('rate', 150)     # setting up new voice rate

"""VOLUME"""
volume = engine.getProperty('volume')   #getting to know current volume level (min=0 and max=1)
engine.setProperty('volume',1.0)    # setting up volume level  between 0 and 1

"""VOICE"""
voices = engine.getProperty('voices')       #getting details of current voice
engine.setProperty('voice', voices[0].id)  #changing index, changes voices. o for male
#engine.setProperty('voice', voices[1].id)   #changing index, changes voices. 1 for female


'''
This section provides tools for adjusting basic settings of the voice recognition system
'''
# Define r as the voice recognizer
recognize = sr.Recognizer()
# Represents the energy level threshold for sounds. Values below this threshold are considered silence, and values above this threshold are considered speech. Can be changed.
recognize.energy_threshold = 50
# Enable or disable dynamic energy threshold
recognize.dynamic_energy_threshold = False
# Represents the minimum length of silence (in seconds) that will register as the end of a phrase. Can be changed.
recognize.pause_threshold = 0.5

#with sr.Microphone() as source:
#    recognize.adjust_for_ambient_noise(source)

# Listen for the wake word
def listening(wake_word):
    recognize.pause_threshold = 0.5
    try:
        with sr.Microphone() as source:
            print("Listening")
            voice = recognize.listen(source,1,1)
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
    recognize.pause_threshold = 2
    try:
        with sr.Microphone() as source:
            print("Getting Command")
            voice = recognize.listen(source,10)
            audio = recognize.recognize_google(voice)
            audio = audio.lower()
    except:
        return "Im sorry, I could not understand that"
    return audio


# Respond to the user
def response(output):
    try:
        engine.say(output)
        engine.runAndWait()
    except sr.UnknownValueError:
        engine.say("Im sorry I could not understand that")
        engine.runAndWait()
    except sr.RequestError as e:
        engine.say("Recognition error; {0}".format(e))
        engine.runAndWait()

