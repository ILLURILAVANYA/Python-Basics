# -*- coding: utf-8 -*-
"""day_5,6 assignments.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1aKJMdKc1jQG-8MSRomiFB14NrOZ4L9tv
"""

from collections import Counter

def calculate_word_frequency(text):
    # Remove punctuation and convert text to lowercase
    text = ''.join(char.lower() if char.isalnum() or char.isspace() else ' ' for char in text)
    # Split the text into words
    words = text.split()
    # Use Counter to count the frequency of each word
    word_count = Counter(words)
    return word_count

def print_word_frequencies(word_count):
    print("Word Frequencies:")
    for word, count in word_count.items():
        print(f"{word}: {count}")

# Input text
text = input("Enter the text: ")

# Calculate and print word frequencies
word_frequencies = calculate_word_frequency(text)
print_word_frequencies(word_frequencies)

import nltk
import spacy
from nltk.corpus import stopwords

# Download NLTK stopwords
nltk.download('stopwords')

def process_text(text):
    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')

    # Convert text to lowercase
    text = text.lower()

    # Tokenize text using spaCy
    doc = nlp(text)

    # Get NLTK stopwords
    stop_words = set(stopwords.words('english'))

    # Remove stopwords
    filtered_words = [token.text for token in doc if token.text not in stop_words]

    return filtered_words

# Input text
text = input("Enter the text: ")

# Process text
filtered_words = process_text(text)

# Print the result
print("Filtered Text (without stopwords):")
print(" ".join(filtered_words))