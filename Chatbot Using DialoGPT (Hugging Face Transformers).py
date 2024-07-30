import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time

# Load the model and tokenizer
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

class ChatBot():
    # Initialize
    def __init__(self):
        # Once chat starts, the history will be stored for chat continuity
        self.chat_history_ids = None
        # Make input ids global to use them anywhere within the object
        self.bot_input_ids = None
        # A flag to check whether to end the conversation
        self.end_chat = False
        # Greet while starting
        self.welcome()
        
    def welcome(self):
        st.write("Initializing BETA ...")
        # Some time to get user ready
        time.sleep(2)
        st.write('Type "bye" or "quit" or "exit" to end chat')
        # Give time to read what has been printed
        time.sleep(3)
        # Greet and introduce
        greeting = np.random.choice([
            "Welcome, I am BETA, here for your kind service",
            "Hey, Great day! I am your virtual assistant",
            "Hello, it's my pleasure meeting you",
            "Hi, I am a BETA. Let's chat!"
        ])
        st.write("BETA >> " + greeting)
        
    def user_input(self, text):
        # Receive input from user
        if text.lower().strip() in ['bye', 'quit', 'exit']:
            # Turn flag on 
            self.end_chat=True
            # A closing comment
            st.write('BETA >> See you soon! Bye!')
            time.sleep(1)
            st.write('Quitting BETA ...')
        else:
            # Continue chat, preprocess input text
            # Encode the new user input, add the eos_token and return a tensor in Pytorch
            self.new_user_input_ids = tokenizer.encode(text + tokenizer.eos_token, return_tensors='pt')

    def bot_response(self):
        # Append the new user input tokens to the chat history if chat has already begun
        if self.chat_history_ids is not None:
            self.bot_input_ids = torch.cat([self.chat_history_ids, self.new_user_input_ids], dim=-1) 
        else:
            # If first entry, initialize bot_input_ids
            self.bot_input_ids = self.new_user_input_ids
        
        # Define the new chat_history_ids based on the preceding chats
        # Generated a response while limiting the total chat history to 1000 tokens
        self.chat_history_ids = model.generate(self.bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)
            
        # Last output tokens from bot
        response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        # In case, bot fails to answer
        if response == "":
            response = self.random_response()
        # Print bot response
        st.write('BETA >> ' + response)
        
    # In case there is no response from model
    def random_response(self):
        i = -1
        response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[i]:][0], skip_special_tokens=True)
        # Iterate over history backwards to find the last token
        while response == '':
            i = i-1
            response = tokenizer.decode(self.chat_history_ids[:, self.bot_input_ids.shape[i]:][0], skip_special_tokens=True)
        # If it is a question, answer suitably
        if response.strip() == '?':
            reply = np.random.choice(["I don't know", "I am not sure"])
        # Not a question? answer suitably
        else:
            reply = np.random.choice(["Great", "Fine. What's up?", "Okay"])
        return reply

# Streamlit app
def main():
    st.title("Chat with BETA")
    
    # Initialize the chatbot
    if 'bot' not in st.session_state:
        st.session_state.bot = ChatBot()
        
    # Input box for user
    user_input = st.text_input("USER >>", key="input")
    
    # Ensure the bot processes user input and generates response only if there is user input
    if user_input:
        st.session_state.bot.user_input(user_input)
        if not st.session_state.bot.end_chat:
            st.session_state.bot.bot_response()

if __name__ == "__main__":
    main()
