# This file manages requests to the Home Assistant Framework

import requests

url = "http://homeassistant.local:8123/api/conversation/process"
headers = {
    "Authorization": "Bearer TOKEN", # Enter your token where it says TOKEN
    "Content-Type": "application/json"
}

def extract_response(data):
    try:
        return data['response']['speech']['plain']['speech']
    except KeyError:
        return None

def execute_command(command):
    data = {
        "text": command,
        "language": "en"
    }

    response = requests.post(url, json=data, headers=headers)

    if response.headers.get('Content-Type') == 'application/json':
        try:
            response_data = response.json()
            speech_text = extract_response(response_data)
            if speech_text:
                print(speech_text)
            else:
                print("Speech response not found")
        except requests.exceptions.JSONDecodeError as e:
            print("JSON decoding failed:", e)
    else:
        print("FAILED: ", response.text)

