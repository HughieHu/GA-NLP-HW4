import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse

import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

def ensure_nltk_resources():
    required_resources = [
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/wordnet', 'wordnet'),
        ('corpora/omw-1.4', 'omw-1.4')
    ]
    
    for resource_path, resource_name in required_resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            print(f"📥 Downloading NLTK resource: {resource_name}...")
            try:
                nltk.download(resource_name, quiet=False)
                print(f"✅ Successfully downloaded: {resource_name}")
            except Exception as e:
                print(f"❌ Failed to download {resource_name}: {e}")
                print(f"   Please manually download using: python -m nltk.downloader {resource_name}")

ensure_nltk_resources()

from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer


random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    keyboard_neighbors = {
        'a': ['q', 's', 'z'],
        'b': ['v', 'g', 'h', 'n'],
        'c': ['x', 'd', 'f', 'v'],
        'd': ['s', 'e', 'f', 'c', 'x'],
        'e': ['w', 'r', 'd', 's'],
        'f': ['d', 'r', 'g', 'v', 'c'],
        'g': ['f', 't', 'h', 'b', 'v'],
        'h': ['g', 'y', 'j', 'n', 'b'],
        'i': ['u', 'o', 'k', 'j'],
        'j': ['h', 'u', 'k', 'n', 'm'],
        'k': ['j', 'i', 'l', 'm'],
        'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k'],
        'n': ['b', 'h', 'j', 'm'],
        'o': ['i', 'p', 'l', 'k'],
        'p': ['o', 'l'],
        'q': ['w', 'a'],
        'r': ['e', 't', 'f', 'd'],
        's': ['a', 'w', 'd', 'x', 'z'],
        't': ['r', 'y', 'g', 'f'],
        'u': ['y', 'i', 'j', 'h'],
        'v': ['c', 'f', 'g', 'b'],
        'w': ['q', 'e', 's', 'a'],
        'x': ['z', 's', 'd', 'c'],
        'y': ['t', 'u', 'h', 'g'],
        'z': ['a', 's', 'x']
    }
    
    def get_synonym(word):
        """Get a synonym for the given word using WordNet"""
        synsets = wordnet.synsets(word)
        if not synsets:
            return None

        lemmas = synsets[0].lemmas()
        synonyms = []
        for lemma in lemmas:
            synonym = lemma.name().replace('_', ' ')
            if synonym.lower() != word.lower():
                synonyms.append(synonym)
        
        if synonyms:
            return random.choice(synonyms)
        return None
    
    def add_keyboard_typo(word):
        """Add keyboard-based typo to a word (10% probability)"""
        if len(word) <= 2: 
            return word
        
        word_list = list(word.lower())
        pos = random.randint(1, len(word_list) - 2)
        char = word_list[pos]

        if char in keyboard_neighbors:
            word_list[pos] = random.choice(keyboard_neighbors[char])
        
        return ''.join(word_list)
    
    def swap_adjacent_chars(word):
        """Swap two adjacent characters in a word (5% probability)"""
        if len(word) <= 2:
            return word
        
        word_list = list(word)
        pos = random.randint(0, len(word_list) - 2)
        word_list[pos], word_list[pos + 1] = word_list[pos + 1], word_list[pos]
        return ''.join(word_list)
    
    def word_swap(tokens):
        """Randomly swap adjacent words (5% of sentences)"""
        if len(tokens) <= 3:
            return tokens
        
        new_tokens = tokens.copy()
        num_swaps = max(1, len(tokens) // 20)  # Swap ~5% of possible positions
        
        for _ in range(num_swaps):
            if len(new_tokens) >= 2:
                pos = random.randint(0, len(new_tokens) - 2)
                new_tokens[pos], new_tokens[pos + 1] = new_tokens[pos + 1], new_tokens[pos]
        
        return new_tokens

    text = example["text"]
    tokens = word_tokenize(text)

    # Apply word swap with 5% probability
    if random.random() < 0.15:
        tokens = word_swap(tokens)

    transformed_tokens = []
    
    # Apply token-level transformations
    for token in tokens:
        if token.isalpha() and len(token) > 2:
            rand = random.random()

            # Synonym replacement 
            if rand < 0.35:
                synonym = get_synonym(token)
                if synonym:
                    transformed_tokens.append(synonym)
                    continue

            # Keyboard typo 
            elif rand < 0.50:
                typo_word = add_keyboard_typo(token)
                transformed_tokens.append(typo_word)
                continue
            
            # Character swap 
            elif rand < 0.60:
                swapped_word = swap_adjacent_chars(token)
                transformed_tokens.append(swapped_word)
                continue
        
        # Keep original token (remaining 75% + non-alphabetic tokens)
        transformed_tokens.append(token)

    detokenizer = TreebankWordDetokenizer()
    example["text"] = detokenizer.detokenize(transformed_tokens)

    ##### YOUR CODE ENDS HERE ######

    return example
