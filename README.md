# AI-CHATBOT-WITH-NLP

**COMPANY** : CODETECH IT SOLUTIONS

**NAME**: Pankaj Singh

**INTERN ID** : CT12LAA

**DOMAIN** : Python Programmimg

**TASK** : Task 3 :- AI Chatbot With NLP

**BATCH DURATION** : January 10th, 2025 to March 10th, 2025

**MENTOR NAME** : Neela Santhosh Kumar

# DESCRIPTION OF THE TASK PERFORMED : AI CHATBOT WITH NLP

1. Objective

The objective of this task is to design and develop an AI chatbot using Natural Language Processing (NLP) libraries, such as NLTK and spaCy, in Python. The chatbot is expected to interpret user queries and respond with accurate, pre-defined answers. This involves creating a training dataset, preprocessing data, training a neural network model, and integrating the model into a conversational interface.

2. Overview of Approach

The chatbot was designed and implemented in multiple phases:

Data Preparation:

Creating an intents.json file containing user query patterns and corresponding responses.

Structuring intents into categories (e.g., greetings, gratitude, farewells).

Preprocessing:

Tokenizing and lemmatizing text data to prepare it for model training.

Ignoring irrelevant characters such as punctuation marks.

Model Training:

Creating a neural network using TensorFlow/Keras to classify user inputs into predefined intents.

Generating bag-of-words (BoW) representations for input queries.

Chatbot Integration:

Writing a Python script to load the trained model, process user input, and generate appropriate responses.

Providing real-time interaction with the chatbot through a simple CLI interface.

3. Implementation Details

3.1 Tools and Libraries Used

Python: Core programming language.

NLTK: Tokenization, lemmatization, and preprocessing.

TensorFlow/Keras: For building and training the neural network model.

NumPy: Handling arrays and numerical data.

Pickle: Saving and loading preprocessed data for efficiency.

JSON: Reading and managing the intents.json file.

3.2 Dataset

The intents.json file was structured as follows:

Intents:

Defined categories such as "greeting," "goodbye," "thanks," and "query."

Each category contained:

Patterns: Sample user inputs (e.g., "Hi there," "Bye").

Responses: Chatbot replies corresponding to each intent.

Example:

{
  "intents": [
    {
      "tag": "greeting",
      "patterns": ["Hi there", "Hello", "Good day"],
      "responses": ["Hello! How can I help?", "Hi there!"]
    }
  ]
}

3.3 Preprocessing Steps

Tokenization: Breaking sentences into individual words using NLTK.

Lemmatization: Reducing words to their base forms (e.g., "running" → "run").

Stopword Removal: Eliminating irrelevant words (e.g., "is," "the," "a").

Creating Bag of Words:

A binary vector representation indicating the presence/absence of words in the vocabulary.

3.4 Model Architecture

Input Layer:

Accepts bag-of-words vectors as input.

Hidden Layers:

Two fully connected layers with ReLU activation.

Dropout layers to prevent overfitting.

Output Layer:

A softmax layer that predicts the intent category.

Loss Function: Categorical Crossentropy.

Optimizer: Stochastic Gradient Descent (SGD) with momentum.

4. Challenges Encountered

4.1 Data Preparation

Issue: Designing a diverse and representative set of patterns and responses for each intent.

Solution: Conducted research to identify common user queries and manually expanded the dataset.

4.2 Unicode Error in File Path

Issue: The intents.json file path caused a UnicodeEscape error.

Solution: Used a raw string (r"path") to resolve the error.

4.3 Low Accuracy in Initial Model

Issue: The neural network's accuracy was low during early training stages.

Solution: Adjusted hyperparameters (e.g., learning rate, batch size) and increased training epochs.

5. Outcomes and Results

Model Accuracy:

Achieved a training accuracy of 95% after fine-tuning.

Functionality:

The chatbot successfully classifies user inputs into intents and responds appropriately.

Example Interaction:

User: Hi there
Bot: Hello! How can I help?

User: Thank you
Bot: My pleasure!

User: What is Codtech IT Solution?
Bot: Codtech IT Solutions Private Limited is a certified company

Scalability:

The chatbot can easily be extended by adding more intents and patterns to intents.json.

6. Future Enhancements

Integrating spaCy:

Use advanced NLP features such as named entity recognition (NER) and dependency parsing.

Contextual Responses:

Implement context-based responses to maintain conversation flow.

Deployment:

Deploy the chatbot as a web application using Flask or Django.

Voice Integration:

Add speech-to-text and text-to-speech capabilities for voice-based interaction.

7. Code and Files

7.1 Training Script

The training script preprocesses data, builds the neural network, and saves the trained model. Key components:

Preprocessing:

Tokenization and lemmatization of patterns.

Creating bag-of-words vectors.

Neural Network:

Input, hidden, and output layers defined using TensorFlow/Keras.

Saving Data:

Pickle files (words.pkl, classes.pkl) store vocabulary and class labels.

Model saved as chatbot_model.h5.

7.2 Chatbot Script

The chatbot script handles:

Loading the trained model.

Processing user input.

Predicting intent and generating responses.

7.3 intents.json

Defines the training dataset with intents, patterns, and responses.

8. Conclusion

This task successfully demonstrated the development of a functional AI chatbot using NLP and Python. The implementation involved preparing data, training a neural network, and integrating the model into a user interface. With minor adjustments and future enhancements, the chatbot can be scaled for real-world applications, such as customer support and personal assistants.
