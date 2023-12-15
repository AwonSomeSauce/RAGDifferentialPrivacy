import gc
import pandas as pd
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
    BertForSequenceClassification,
)
from mechanisms.detectors.santext_detector import SanTextDetector
from mechanisms.detectors.custext_detector import CusTextDetector
from mechanisms.detectors.presidio_detector import PresidioDetector
from mechanisms.custext import CusText
from mechanisms.santext import SanText
from evaluators.training import Trainer
from evaluators.utils import Bert_dataset

# Constants
WORD_EMBEDDING = "glove"
WORD_EMBEDDING_PATH = "glove.42B.300d.txt"
TOP_K = 20
P = 0.3
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
EPS = 1e-8
BERT_MODEL = "bert-base-uncased"
DATASET = "qnli"


def preprocess_text(text):
    """Remove specific substrings and strip text."""
    text = text.replace("qnli question:", "")
    text = text.replace("sentence:", "")
    return text.strip()


def preprocess_data(df):
    # Rename 'text' column to 'sentence' and preprocess text
    df = df.rename(columns={"text": "sentence"})
    df["sentence"] = df["sentence"].apply(preprocess_text)

    # Convert sentences to lowercase
    df["sentence"] = df["sentence"].apply(lambda x: x.lower())

    # Convert 'label' values from 'not_entailment'/'entailment' to 0/1
    label_mapping = {"not_entailment": 0, "entailment": 1}
    df["label"] = df["label"].replace(label_mapping)

    return df


def postprocess_df(df):
    df = df.drop("sentence", axis=1, inplace=True)
    df = df.rename(columns={"sanitized sentence": "sentence"}, inplace=True)

    return df


def load_and_prepare_data():
    huggingface_name = "carlosejimenez/seq2seq-qnli" if DATASET == "qnli" else DATASET
    dataset = load_dataset(huggingface_name)

    # Determine the column to drop based on the dataset name
    col_to_drop = "orig_idx" if DATASET == "qnli" else "idx"

    # Process each split
    def process_split(split):
        df = (
            dataset[split]
            .to_pandas()
            .drop(columns=[col_to_drop])
            .reset_index(drop=True)
        )
        return preprocess_data(df) if DATASET == "qnli" else df

    return {
        "train": process_split("train"),
        "validation": process_split("validation"),
    }


def create_mechanism(detector, epsilon):
    """Create a mechanism instance based on the detector type."""
    if detector["mechanism"] == SanText:
        return detector["mechanism"](
            WORD_EMBEDDING, WORD_EMBEDDING_PATH, epsilon, P, detector["detector"]
        )
    elif detector["mechanism"] == CusText:
        return detector["mechanism"](
            WORD_EMBEDDING, WORD_EMBEDDING_PATH, epsilon, TOP_K, detector["detector"]
        )
    else:
        raise ValueError("Unknown mechanism type provided.")


def initialize_model_and_optimizer():
    """Initialize the BERT model and the optimizer."""
    model = BertForSequenceClassification.from_pretrained(
        BERT_MODEL, num_labels=2, output_attentions=False, output_hidden_states=False
    )
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, eps=EPS)
    return model, optimizer


def train_and_evaluate(train_df, validation_df, model, optimizer):
    """Train the model and evaluate on the validation set."""
    train_dataset = Bert_dataset(train_df)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataset = Bert_dataset(validation_df)
    validation_loader = DataLoader(
        validation_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_loader) * 3
    )

    trainer = Trainer(
        model,
        scheduler,
        optimizer,
        3,  # Number of epochs
        50,  # Logging steps
        50,  # Evaluation steps
        True,
        None,
    )
    trainer.train(train_loader, validation_loader)
    return trainer.predict(validation_loader)


def main():
    """Main function to run the training and evaluation."""
    data = load_and_prepare_data()
    detectors = {
        "SanText": {"mechanism": SanText, "detector": SanTextDetector(0.9)},
        "CusText": {"mechanism": CusText, "detector": CusTextDetector()},
        "SanText + Presidio": {"mechanism": SanText, "detector": PresidioDetector()},
        "CusText + Presidio": {"mechanism": CusText, "detector": PresidioDetector()},
    }
    epsilons = [1.0, 2.0, 3.0]

    results_df = pd.DataFrame(columns=["Mechanism", "Accuracy"])

    model, optimizer = initialize_model_and_optimizer()

    # Training and evaluation with the original data
    accuracy = train_and_evaluate(data["train"], data["validation"], model, optimizer)
    results_row = {"Mechanism": "Original", "Accuracy": accuracy}
    results_df = pd.concat([results_df, pd.DataFrame([results_row])], ignore_index=True)

    # Training and evaluation with sanitized data
    for epsilon in epsilons:
        for name, detector in detectors.items():
            mechanism = create_mechanism(detector, epsilon)
            mechanism_name = f"{name} with epsilon = {epsilon}"
            train_data = mechanism.sanitize(data["train"])
            validation_data = mechanism.sanitize(data["validation"])
            train_data.head(100).to_csv(
                f"{mechanism_name} sanitized {DATASET}.csv", index=False
            )
            train_data = postprocess_df(train_data)
            validation_data = postprocess_df(validation_data)
            accuracy = train_and_evaluate(train_data, validation_data, model, optimizer)
            results_row = {"Mechanism": mechanism_name, "Accuracy": accuracy}
            results_df = pd.concat(
                [results_df, pd.DataFrame([results_row])], ignore_index=True
            )

            # Free memory
            del mechanism, train_data, validation_data
            gc.collect()

    # After all evaluations, save the results to a CSV file
    results_df.to_csv("results_qnli.csv", index=False)


if __name__ == "__main__":
    main()
