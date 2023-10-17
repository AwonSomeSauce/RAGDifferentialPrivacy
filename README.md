# RAG Differential Privacy

This repository focuses on enhancing the Retriever-Augmented Generation (RAG) data with robust differential privacy measures, ensuring both data security and user privacy.

## Overview

The Retriever-Augmented Generation model, or RAG, combines the powers of large-scale retrievers and seq2seq models. However, as with most data-driven models, there's a concern for user privacy. This project is our effort to imbue RAG data with differential privacy mechanisms to protect individual data points, without compromising the efficacy of the model.

## Getting Started

### 1. Collecting GloVe Word Embeddings

Before diving into the project, you'll need the GloVe word embeddings. They play a crucial role in certain data processing steps and ensuring the data's differential privacy.

#### Instructions:

- Go to the [GloVe project page](https://nlp.stanford.edu/projects/glove/)
- Download the `glove.840B.300d.zip` embeddings of your choice.
- Once downloaded, save the embeddings in the root directory of this repository.

### 2. Installing the Presidio Library

Presidio is a library developed by Microsoft to recognize, analyze, and redact personally identifiable information (PII) and other sensitive data in text. For this project, it's crucial to have the necessary Presidio components set up.

#### Instructions:

Install the required Presidio components using pip:

```bash
pip install presidio_analyzer presidio_anonymizer
```

To ensure the Presidio analyzers work effectively, you should also download the `en_core_web_lg` model for spaCy:

```bash
python -m spacy download en_core_web_lg
```
