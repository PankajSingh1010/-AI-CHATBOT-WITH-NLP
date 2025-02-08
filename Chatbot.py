# Importing libraries
import random
import json
import pickle
import numpy as np
import nltk

from nltk.stem import WordNetLemmatizer
from keras.models import load_model

# Initialize the WordNet Lemmatizer to reduce words to their root form.
lemmatizer = WordNetLemmatizer()

# Load the intents file, which contains patterns, responses, and intent tags.
intents = json.loads(open(r'C:\Users\ASUS\Downloads\Python Internship\Chatbot\intents.json').read())

# Load preprocessed data (word list and class list) from pickle files.
words = pickle.load(open('words.pkl', 'rb'))  # List of unique lemmatized words.
classes = pickle.load(open('classes.pkl', 'rb'))  # List of unique intent classes.

# Load the trained model that was previously saved.
model = load_model('chatbot_model.h5')

# Function to clean up a given sentence by tokenizing and lemmatizing it.
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)  # Tokenize the input sentence into words.
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]  # Lemmatize each word.
    return sentence_words

# Function to create a bag of words representation for a given sentence.
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)  # Tokenize and lemmatize the sentence.
    bag = [0] * len(words)  # Initialize a list of zeros equal to the size of the word list.
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:  # If the word exists in the predefined word list, set its index to 1.
                bag[i] = 1
    return np.array(bag)  # Return the bag of words as a NumPy array.

# Function to predict the intent class of a given input sentence.
def predict_class(sentence):
    bow = bag_of_words(sentence)  # Convert the sentence to a bag of words.
    res = model.predict(np.array([bow]))[0]  # Predict probabilities for each class using the model.
    ERROR_THRESHOLD = 0.25  # Define a threshold to filter out low-confidence predictions.
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]  # Filter predictions above the threshold.

    results.sort(key=lambda x: x[1], reverse=True)  # Sort results by probability in descending order.
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})  # Map predictions to intent classes.
    return return_list

# Function to get a response based on the predicted intent.
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']  # Get the intent with the highest probability.
    list_of_intents = intents_json['intents']  # Retrieve the list of intents from the intents JSON.
    for i in list_of_intents:
        if i['tag'] == tag:  # Match the tag with the corresponding intent in the JSON file.
            result = random.choice(i['responses'])  # Select a random response from the matched intent.
            break
    return result

# Inform the user that the chatbot is ready to use.
print("Bot is running! Let's go!")

# Infinite loop to continuously receive user input and provide responses.
while True:
    message = input("")  # Take user input.
    ints = predict_class(message)  # Predict the intent class of the user input.
    res = get_response(ints, intents)  # Get an appropriate response for the predicted intent.
    print(res)  # Print the chatbot's response.
