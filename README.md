# Atomic-GPT

![atomicGPT](./assets/box.jpg)

A simple implementation medium sized GPT⚡, a rewrite of nanoGPT by Andrej Karpathy🤠

This repository is dedicated to implementing and experimenting with two different language models: a Bigram Language Model (`bigram.py`) and a more advanced GPT (Generative Pre-trained Transformer) model (`gpt.py`). The repository also includes a training script (`train.py`) to train and evaluate the GPT model. Additionally, there is a `utils` directory containing essential utility files for data loading and tokenization.

## Components

### 1. Bigram Language Model (`bigram.py`)

The Bigram Language Model is implemented using PyTorch and focuses on predicting the next token in a sequence based on bigram relationships. The file includes hyperparameters, data loading, and the `BigramLanguageModel` class.

### 2. GPT Model (`gpt.py`)

The GPT model is a more sophisticated language model based on the Transformer architecture. It incorporates multi-head self-attention mechanisms and layer normalization. The file consists of classes such as `LayerNorm1d`, `Head`, `MultiHeadAttention`, `FeedFoward`, `Block`, and `GPTLanguageModel`, each contributing to the overall structure of the GPT model.

### 3. Training Script (`train.py`)

The training script (`train.py`) uses the GPT model to train on a given dataset. It initializes the GPT model, performs training iterations, and periodically evaluates the model on both training and validation sets. The script also includes functionality to generate text from the trained model.

### 4. Utility Files (`utils/`)

#### a. Tokenizer (`tokenizer.py`)

The `CharTokenizer` class in `tokenizer.py` facilitates the encoding and decoding of strings into lists of integers and vice versa. This utility is essential for preprocessing text data before training the language models.

#### b. Data Loader (`dataloader.py`)

The `dataloader.py` file contains a function for loading data from a specified file path. This function reads the data, identifies unique characters, and returns both the raw data and a list of unique characters.

## Usage

1. **Bigram Language Model (`bigram.py`):**
   - Load data from `data/tinyshakespeare.txt`.
   - Train the Bigram Language Model.

2. **GPT Model (`gpt.py`):**
   - Load data from `data/tinyshakespeare.txt`.
   - Train the GPT model using the training script (`train.py`).

3. **Training Script (`train.py`):**
   - Train and evaluate the GPT model on the provided dataset.
   - Monitor training and validation losses.

4. **Utilities (`utils/`):**
   - Utilize the `CharTokenizer` for encoding and decoding strings.
   - Load data using the `char_load_data` function.

Feel free to explore, experiment, and extend the functionalities of the language models within this repository!
