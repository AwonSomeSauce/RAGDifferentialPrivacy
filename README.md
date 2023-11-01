# RAG Differential Privacy

This repository focuses on enhancing the Retriever-Augmented Generation (RAG) data with robust differential privacy measures, ensuring both data security and user privacy.

## Overview

The Retriever-Augmented Generation model, or RAG, combines the powers of large-scale retrievers and seq2seq models. However, as with most data-driven models, there's a concern for user privacy. This project is our effort to imbue RAG data with differential privacy mechanisms to protect individual data points, without compromising the efficacy of the model.

## Getting Started

### Prerequisite: Setting Up a Virtual Environment

To avoid library conflicts, it's recommended to use a virtual environment.

```bash
# Install virtualenv if not already
pip install virtualenv

# Create a virtual environment named 'venv'
virtualenv venv

# Activate the virtual environment
# Windows:
venv\Scripts\activate
# macOS and Linux:
source venv/bin/activate
```

### 0. Installing Necessary Libraries

```bash
pip install pandas numpy tqdm spacy sklearn scipy nltk presidio_analyzer presidio_anonymizer
python -m spacy download en_core_web_lg
```

### 1. Collecting GloVe Word Embeddings

Before diving into the project, you'll need the GloVe word embeddings. They play a crucial role in certain data processing steps and ensuring the data's differential privacy.

#### Instructions:

- Go to the [GloVe project page](https://nlp.stanford.edu/projects/glove/)
- Download the `glove.840B.300d.zip` embeddings of your choice.
- Once downloaded, save the embeddings in the root directory of this repository.

### Instructions:

In your root folder, run:

```bash
python main.py
```
