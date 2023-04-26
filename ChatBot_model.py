import random

# Define possible user inputs and their corresponding bot responses
responses = {
    "hello": ["Hi there!", "Hello!", "Hey!"],
    "how are you": ["I'm doing well, thank you.", "Not too bad, thanks for asking.", "I'm just fine."],
    "what's your name": ["My name is Chatbot.", "I go by Chatbot.", "You can call me Chatbot."],
    "default": ["I'm sorry, I didn't understand what you said.", "Can you please rephrase that?", "I'm not sure what you mean."],
}

# Define a function to get a response from the bot
def get_response(user_input):
    # Convert the user input to lowercase and remove whitespace
    user_input = user_input.lower().strip()
    
    # Check if the user input matches any of the defined responses
    if user_input in responses:
        return random.choice(responses[user_input])
    else:
        return random.choice(responses["default"])

# Define the main function to run the chatbot
def run_chatbot():
    # Greet the user
    print("Hi, I'm Chatbot. How can I help you today?")
    
    # Loop through user input and bot response
    while True:
        # Get user input
        user_input = input("You: ")
        
        # Get bot response
        bot_response = get_response(user_input)
        
        # Print bot response
        print("Chatbot: " + bot_response)
        
        # Check if user wants to end the conversation
        if user_input.lower().strip() == "bye":
            print("Chatbot: Goodbye!")
            break

# Run the chatbot
run_chatbot()
