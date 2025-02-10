# Text-Prediction & Generation 📝

## Introduction
In this project, I trained a machine learning model using an LSTM-based neural network to predict and generate text sequences. The model is trained on a dataset of news articles, and while the generated text may not be perfect, it demonstrates the principles of sequence modeling and natural language processing.

Below is a step-by-step explanation of the commands used in this project, showing how the model processes text data and generates predictions.

---

## 1. Importing Libraries 📦

```python
import random
import pickle

import numpy as np
import pandas as pd
from nltk.tokenize import RegexpTokenizer

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop
```
- **random**: Used for randomly selecting elements, particularly during text generation.
- **pickle**: Used for serializing and saving Python objects (not explicitly used in this snippet).
- **numpy (np)**: Handles numerical operations and array manipulations efficiently.
- **pandas (pd)**: Used for handling structured data in DataFrame format.
- **nltk.tokenize.RegexpTokenizer**: Splits text into words while removing punctuation and non-alphanumeric characters.
- **tensorflow.keras**: Provides deep learning tools for building and training the neural network:
  - **Sequential**: Creates a linear stack of layers for the neural network.
  - **load_model**: Loads a pre-trained Keras model.
  - **LSTM**: A recurrent neural network (RNN) layer for processing sequential text data.
  - **Dense**: Fully connected layer for classification.
  - **Activation**: Applies activation functions to neuron outputs.
  - **RMSprop**: An optimizer that helps adjust model weights efficiently.

---

## 2. Loading and Exploring the Dataset 🗃

```python
text_data_frame = pd.read_csv("fake_or_real_news.csv")
text_data_frame
```
- **pd.read_csv("fake_or_real_news.csv")**: Loads the dataset containing news articles.
- **text_data_frame**: Displays the dataset for inspection in Jupyter Notebook.

---

## 3. Preprocessing Text 📝

```python
text = list(text_data_frame.text.values)
joined_text = " ".join(text)

partial_text = joined_text[:10000]
```
- **text = list(text_data_frame.text.values)**: Extracts the text column as a list of article contents.
- **joined_text = " ".join(text)**: Combines all articles into a single string.
- **partial_text = joined_text[:10000]**: Uses only the first 10,000 characters for training (reducing computation requirements).

---

## 4. Tokenization ⚙️

```python
tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(partial_text.lower())
```
- **RegexpTokenizer(r"\w+")**: Tokenizes words while ignoring punctuation and special characters.
- **tokenizer.tokenize(partial_text.lower())**: Converts text to lowercase and tokenizes it into individual words.

---

## 5. Identifying Unique Tokens & Indexing 🔢

```python
unique_tokens = np.unique(tokens)
unique_token_index = {token: idx for idx, token in enumerate(unique_tokens)}
```
- **np.unique(tokens)**: Finds unique words in the dataset.
- **unique_token_index**: Creates a dictionary mapping each unique token to an index for encoding.

---

## 6. Creating Training Sequences 🚀

```python
number_of_words = 10
input_words = []
next_words = []

for i in range(len(tokens) - number_of_words):
    input_words.append(tokens[i:i + number_of_words])
    next_words.append(tokens[i + number_of_words])
```
- **number_of_words = 10**: Defines sequence length (how many words the model looks at before predicting the next word).
- **input_words / next_words**: Store training data.
- **Loop**: Creates sequences of 10 words (**input_words**) and the next word (**next_words**).

---

## 7. One-Hot Encoding the Sequences 🌈

```python
x = np.zeros((len(input_words), number_of_words, len(unique_tokens)), dtype=bool)
y = np.zeros((len(next_words), len(unique_tokens)), dtype=bool)

for i, words in enumerate(input_words):
    for j, word in enumerate(words):
        x[i, j, unique_token_index[word]] = 1
    y[i, unique_token_index[next_words[i]]] = 1
```
- **x / y**: Create arrays to store training data.
- **One-hot encoding**: Converts words into binary vectors where only one position is `1` (word index) and the rest are `0`.

---

## 8. Building and Training the Model 🏗

```python
model = Sequential()
model.add(LSTM(128, input_shape=(number_of_words, len(unique_tokens)), return_sequences=True))
model.add(LSTM(128))
model.add(Dense(len(unique_tokens), activation='softmax'))
```
- **Sequential model**: Stack of layers.
- **LSTM layers**: Process sequential text data.
- **Dense layer with softmax**: Outputs probabilities for each word.

```python
model.compile(loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.01), metrics=["accuracy"])
model.fit(x, y, batch_size=128, epochs=30, shuffle=True)
```
- **Categorical cross-entropy loss**: Suitable for multi-class classification.
- **RMSprop optimizer**: Optimizes weights.
- **Trains model for 30 epochs**.

---

## 9. Predicting the Next Word 🔮

```python
def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    x = np.zeros((1, number_of_words, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        x[0, i, unique_token_index[word]] = 1

    predictions = model.predict(x)[0]
    return np.argpartition(predictions, n_best)[-n_best:]
```
- Converts input text to a one-hot encoded array.
- Uses the model to predict the **top n_best words**.

---

## 10. Generating Text 🎉

```python
def generate_text(input_text, text_length, creativity=3):
    word_sequence = input_text.split()
    current = 0
    for _ in range(text_length):
        sub_sequence = " ".join(tokenizer.tokenize(" ".join(word_sequence).lower())[current:current+number_of_words])
        try:
            choice = unique_tokens[random.choice(predict_next_word(sub_sequence, creativity))]
        except:
            choice = random.choice(unique_tokens)
        word_sequence.append(choice)
        current += 1
    return " ".join(word_sequence)
```
- Predicts and generates text using random selections from the top `creativity` choices.

```python
generate_text("He will have to look at this thing and he", 100, 5)
```
- Example of generated text.

---

## Conclusion ✨
This project demonstrates how to preprocess text data, train an LSTM-based neural network, and generate new text sequences. Though the generated output may not be perfect due to limited data, results can be improved by training on larger datasets and fine-tuning hyperparameters.