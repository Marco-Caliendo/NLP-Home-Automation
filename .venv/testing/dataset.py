# dataset.py

train_sentences = [
    "turn on the light in the living room",
    "can you turn on the fan in the bedroom?",
    "switch on the air conditioner in the office",
    "turn on the heater in the bathroom",
    "power on the light in the kitchen",

    "turn off the light in the bedroom",
    "switch off the fan in the living room",
    "can you turn off the air conditioner in the office?",
    "power down the heater in the bathroom",
    "turn off the fan in the kitchen",

    "is the light on in the living room?",
    "is the fan off in the bedroom?",
    "is the air conditioner running in the office?",
    "is the heater on in the bathroom?",
    "is the light off in the kitchen?",

    "set the thermostat to 72 degrees in the living room",
    "increase the brightness of the light in the bedroom",
    "dim the light in the kitchen",
    "set the fan speed to high in the office",
    "set the air conditioner to 68 degrees in the living room",

    "what is the temperature in the living room?",
    "what's the current brightness in the bedroom?",
    "how warm is it in the bathroom?",
    "what's the fan speed in the office?",
    "what's the air quality in the living room?",

    "open the garage door",
    "lock the front door",
    "unlock the back door",
    "open the blinds in the living room",
    "close the blinds in the kitchen",

    "can you switch on the heater in the bathroom?",
    "please turn on the light in the hallway",
    "switch off the fan in the guest room",
    "could you dim the lights in the dining room?",
    "turn off the TV in the living room",

    "is the light in the hallway on?",
    "is the TV off in the living room?",
    "are the windows open in the bedroom?",
    "is the thermostat set to 70 in the kitchen?",
    "are the lights dimmed in the dining room?",

    "turn off the lights in the house at 10 PM",
    "switch on the fan in the bedroom for 2 hours",
    "set the heater to turn off in 30 minutes",
    "schedule the air conditioner to turn on at 8 AM",
    "turn off the TV after this show ends"
]

train_intents = [
    0, 0, 0, 0, 0,  # TurnOnDevice
    1, 1, 1, 1, 1,  # TurnOffDevice
    2, 2, 2, 2, 2,  # QueryDeviceState
    3, 3, 3, 3, 3,  # SetDevice
    4, 4, 4, 4, 4,  # QueryEnvironmentalState
    5, 5, 5, 5, 5,  # ControlMisc
    0, 0, 1, 1, 1,  # More variations
    2, 2, 2, 2, 2,  # More state queries
    6, 6, 6, 6, 6   # TimeBasedControl
]

vocab = {
    'turn': 0, 'on': 1, 'off': 2, 'light': 3, 'in': 4, 'the': 5, 'living': 6, 'room': 7, 'bedroom': 8,
    'kitchen': 9, 'bathroom': 10, 'fan': 11, 'air': 12, 'conditioner': 13, 'thermostat': 14, 'set': 15,
    'to': 16, 'degrees': 17, 'increase': 18, 'brightness': 19, 'dim': 20, 'speed': 21, 'high': 22,
    'is': 23, 'running': 24, 'what': 25, 'temperature': 26, 'garage': 27, 'door': 28, 'open': 29,
    'close': 30, 'windows': 31, 'lock': 32, 'unlock': 33, 'blinds': 34, 'tv': 35, 'turning': 36,
    'on?': 37, 'off?': 38, 'after': 39, 'hours': 40, 'schedule': 41, 'at': 42, 'pm': 43, 'am': 44,
    '<PAD>': 45, '<UNK>': 46
}

intent_map = {
    'TurnOnDevice': 0,
    'TurnOffDevice': 1,
    'QueryDeviceState': 2,
    'SetDevice': 3,
    'QueryEnvironmentalState': 4,
    'ControlMisc': 5,
    'TimeBasedControl': 6
}
