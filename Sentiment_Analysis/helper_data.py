import os
import time
import pandas as pd

import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import random

from googletrans import Translator
import concurrent.futures
from transformers import pipeline
from sklearn.utils import shuffle


nltk.download('wordnet')
nltk.download('stopwords')

'''
Functions for text augmentation:
1. synonym replacement
2. back translation
3. paraphrase
4. oversampling and undersampoing
'''

# 1. Synonym Replacement
def synonym_replacement(sentence, n=1):  # number of word replacements in a sentence
    words = sentence.split()
    new_words = words.copy()
    random_word_list = list(set([word for word in words if word not in stopwords.words("english")]))
    random.shuffle(random_word_list)
    num_replaced = 0

    for random_word in random_word_list:
        synonyms = get_synonyms(random_word)
        if len(synonyms) >= 1:
            synonym = random.choice(synonyms)
            new_words = [synonym if word == random_word else word for word in new_words]
            num_replaced += 1

        if num_replaced >= n:  # Only replace up to n words
            break

    sentence = ' '.join(new_words)

    return sentence

# Helper function to get synonyms for a word
def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return synonyms

# Function for text augmentation
def augment_text_with_synonyms(dataset, n=1):
    augmented_data = []
    for _, row in dataset.iterrows():
        sentence = row['text']
        augmented_sentence = synonym_replacement(sentence, n)
        augmented_data.append((augmented_sentence, row['target']))
    
    augmented_df = pd.DataFrame(augmented_data, columns=['text', 'target'])
    augmented_dataset = pd.concat([dataset, augmented_df], ignore_index=True)
    
    return augmented_dataset


# 2. Back Translation
# This technique involves translating the text to another language and then translating it back to the original language.
def parallel_back_translation(sentence, n=1):
    translator = Translator()
    new_sentences = []

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []

        for _ in range(n):
            # Translate the sentence to a random language
            random_language = random.choice(['fr', 'es', 'de', 'ja', 'ko'])  # You can add more languages
            future = executor.submit(translator.translate, sentence, dest=random_language)
            futures.append(future)

        for future in futures:
            try:
                translated = future.result()
                # Translate the translated sentence back to the original language
                back_translated = translator.translate(translated.text, dest='en').text
                new_sentences.append(back_translated)
            except Exception as e:
                print(f"Error in back translation: {e}")

    return new_sentences

# Function for text augmentation using parallel back translation
def augment_text_with_parallel_back_translation(dataset, n=1):
    augmented_data = []
    for _, row in dataset.iterrows():
        sentence = row['text']
        augmented_sentences = parallel_back_translation(sentence, n)
        for augmented_sentence in augmented_sentences:
            augmented_data.append((augmented_sentence, row['target']))

    augmented_df = pd.DataFrame(augmented_data, columns=['text', 'target'])
    augmented_dataset = pd.concat([dataset, augmented_df], ignore_index=True)
    
    return augmented_dataset


# 3. Paraphrase
def paraphrase_text(sentence, n=1):
    generator = pipeline("text2text-generation", model="tuner007/pegasus_paraphrase")
    new_sentences = []

    for _ in range(n):
        paraphrase = generator(sentence, max_length=60, do_sample=True, top_k=50)[0]['generated_text']
        new_sentences.append(paraphrase)

    return new_sentences

# Function for text augmentation using paraphrasing
def augment_text_with_paraphrasing(dataset, n=1):
    augmented_data = []
    for _, row in dataset.iterrows():
        sentence = row['text']
        augmented_sentences = paraphrase_text(sentence, n)
        for augmented_sentence in augmented_sentences:
            augmented_data.append((augmented_sentence, row['target']))
    
    augmented_df = pd.DataFrame(augmented_data, columns=['text', 'target'])
    augmented_dataset = pd.concat([dataset, augmented_df], ignore_index=True)
    
    return augmented_dataset


# 4. Bootstrapping
def resample_data(dataset, n=1):
    augmented_data = []
    
    for _ in range(n):
        # Randomly select samples with replacement from the original dataset
        resampled_samples = dataset.sample(frac=1, replace=True)  # Replace=True enables bootstrapping
        augmented_data.append(resampled_samples)  # Append the entire DataFrame
        
    # Create a new DataFrame by concatenating the resampled data
    resampled_df = pd.concat(augmented_data, ignore_index=True)
    
    return resampled_df



# 5. oversampling and undersampoing to balance classes
def oversample_data(dataset, n=1):
    augmented_data = []

    # Determine the majority and minority class
    majority_class = dataset[dataset['target'] == 1]
    minority_class = dataset[dataset['target'] == 0]

    for _ in range(n):
        # Randomly select samples from the minority class with replacement
        resampled_samples = minority_class.sample(frac=1, replace=True)
        augmented_data.append(resampled_samples)

    # Concatenate the resampled data with the majority class
    balanced_df = pd.concat([dataset, *augmented_data], ignore_index=True)

    return balanced_df


def undersample_data(dataset, n=1):
    augmented_data = []

    # Determine the majority and minority class
    majority_class = dataset[dataset['target'] == 1]
    minority_class = dataset[dataset['target'] == 0]

    for _ in range(n):
        # Randomly select a subset of samples from the majority class
        undersampled_samples = majority_class.sample(n=len(minority_class), replace=False)
        augmented_data.append(undersampled_samples)

    # Concatenate the undersampled data with the minority class
    balanced_df = pd.concat([minority_class, *augmented_data], ignore_index=True)

    return balanced_df


def shuffle_dataframe(df):
    """
    Shuffle the rows of a DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame to be shuffled.

    Returns:
    pd.DataFrame: A new DataFrame with shuffled rows.
    """
    shuffled_df = shuffle(df)
    return shuffled_df
