# Importing libraries
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

# Initialize the lemmatizer to reduce words to their base form
lemmatizer = WordNetLemmatizer()

# Load the intents file, which contains the chatbot intents and patterns
intents = json.loads(open(r'C:\Users\ASUS\Downloads\Python Internship\Chatbot\intents.json').read())

# Initialize empty lists to store words, classes (tags), and documents (pairs of patterns and tags)
words = []
classes = []
documents = []

# List of characters to ignore when tokenizing words
ignoreLetters = ['?', '!', '.', ',']

# Loop through each intent in the intents file
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each word in the pattern (sentence)
        wordList = nltk.word_tokenize(pattern)
        words.extend(wordList)  # Add all the words to the words list
        documents.append((wordList, intent['tag']))  # Store the pattern and associated tag
        if intent['tag'] not in classes:
            classes.append(intent['tag'])  # Add the tag (intent) to the classes list if it's not already there

# Lemmatize each word (reduce to its base form) and remove any unwanted characters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignoreLetters]
words = sorted(set(words))  # Remove duplicates and sort the words

# Sort and remove duplicates from classes list
classes = sorted(set(classes))

# Save the words and classes lists to files using pickle for future use
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare the training set
training = []
outputEmpty = [0] * len(classes)  # Empty output array that will be used for one-hot encoding the tags

# Loop through each document to create a bag of words and a corresponding output label
for document in documents:
    bag = []
    wordPatterns = document[0]
    wordPatterns = [lemmatizer.lemmatize(word.lower()) for word in wordPatterns]  # Lemmatize and convert to lowercase
    
    # Create a bag of words: 1 if the word is in the pattern, 0 otherwise
    for word in words:
        bag.append(1) if word in wordPatterns else bag.append(0)

    # One-hot encode the output (tag)
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1  # Set 1 for the correct class (tag)
    training.append(bag + outputRow)  # Combine bag of words and the one-hot output

# Shuffle the training data to ensure randomization
random.shuffle(training)
training = np.array(training)

# Split the data into inputs (X) and outputs (Y)
trainX = training[:, :len(words)]  # Inputs: bag of words
trainY = training[:, len(words):]  # Outputs: one-hot encoded tags

# Build the model using Keras Sequential API
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu'))  # First hidden layer
model.add(tf.keras.layers.Dropout(0.5))  # Dropout layer for regularization
model.add(tf.keras.layers.Dense(64, activation='relu'))  # Second hidden layer
model.add(tf.keras.layers.Dropout(0.5))  # Dropout layer for regularization
model.add(tf.keras.layers.Dense(len(trainY[0]), activation='softmax'))  # Output layer with softmax activation

# Use SGD (Stochastic Gradient Descent) optimizer
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])  # Compile the model

# Train the model using the training data
hist = model.fit(np.array(trainX), np.array(trainY), epochs=200, batch_size=5, verbose=1)

# Save the trained model to a file for later use
model.save('chatbot_model.h5', hist)
print('Excecuted Successfully !!')  # Indicate that the training and model saving are complete
