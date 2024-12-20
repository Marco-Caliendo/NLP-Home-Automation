#!/usr/bin/env python3

import os
from openai import OpenAI
from openai.resources.audio import speech
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the environment variable for the openAI API key
gpt_key = 'INPUT KEY HERE'
#gpt_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"),)
gpt_client = OpenAI(api_key=gpt_key, )

tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large", )

# Set the pad_token to something different than eos_token if they are the same
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token to eos_token if not defined

first_input = True

def red_ai(speech):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(speech + tokenizer.eos_token, return_tensors='pt')

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if first_input == False else new_user_input_ids
    #bot_input_ids = new_user_input_ids

    # Create the attention mask (1 for tokens to attend to, 0 for padding)
    attention_mask = torch.ones(bot_input_ids.shape, dtype=torch.long)

    # Generate a response while limiting the total chat history to 1000 tokens
    chat_history_ids = model.generate(
        bot_input_ids,
        max_length=1000,
        pad_token_id=tokenizer.pad_token_id,  # Use the pad_token_id explicitly
        attention_mask=attention_mask
    )

    # Print last output tokens from bot
    bot_response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    print("Bot: {}".format(bot_response))


def chat_gpt(speech):

    chat_completion = gpt_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": speech,
            }
        ],
        model="gpt-3.5-turbo",
    )
    print(f"Assistant: {completion.choices[0].message.content}")



chat_gpt("Say 'This is a test'")
