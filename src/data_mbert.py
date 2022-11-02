import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

from config import PARAGRAPHS, SENTENCES, SEED

np.random.seed(SEED)


def prepare_readability_data():
    paragraphs_df = pd.DataFrame()
    sentences_df = pd.DataFrame()
    for p in PARAGRAPHS:
        df = pd.read_json(p, lines=True)
        df["source"] = Path(p).name
        paragraphs_df = pd.concat([paragraphs_df, df])

    for s in SENTENCES:
        df = pd.read_json(s, lines=True)
        df["source"] = Path(s).name
        sentences_df = pd.concat([sentences_df, df])

    paragraphs_2class_df = paragraphs_df[["text", "level", "source"]]
    paragraphs_2class_df = paragraphs_2class_df.rename(columns={"level": "label"})

    paragraphs_3class_df = paragraphs_df[["text", "level-3", "source"]]
    paragraphs_3class_df = paragraphs_3class_df.rename(columns={"level-3": "label"})

    sentences_2class_df = sentences_df[["text", "level", "source"]]
    sentences_2class_df = sentences_2class_df.rename(columns={"level": "label"})

    sentences_3class_df = sentences_df[["text", "level-3", "source"]]
    sentences_3class_df = sentences_3class_df.rename(columns={"level-3": "label"})

    mappings = {
        "para_2class": paragraphs_2class_df,
        "para_3class": paragraphs_3class_df,
        "sent_2class": sentences_2class_df,
        "sent_3class": sentences_3class_df
    }

    labels = {
        "complex": 1,
        "simple": 0,
        "advanced": 2,
        "intermediate": 1,
        "basic": 0,
    }

    for key, df in mappings.items():
        df = df[df["label"] != "N/A"]
        df["label"] = df["label"].apply(lambda v: labels[v])
        df["text"] = df["text"].apply(lambda v: v.replace("\n", ""))
        prepare_splits(key, "", df=df)


def prepare_splits(dataset, data_file, df=None):
    if df is None:
        df = pd.read_json(data_file, lines=True)

    # Check labels
    print(f"Dataset: {dataset}")
    print(df["label"].unique())
    print(df.value_counts("label"))

    shuffle_df = df.sample(frac=1, random_state=SEED)
    train, valid, test = np.split(shuffle_df, [int(.8 * len(shuffle_df)), int(.9 * len(shuffle_df))])
    print("Train labels:", train.value_counts("label"))
    print("Valid labels:", valid.value_counts("label"))
    print("Test labels:", test.value_counts("label"))

    dataset_dir = f"data/csv/{dataset}"
    os.makedirs(dataset_dir, exist_ok=True)
    train.to_csv(f"{dataset_dir}/{dataset}_train.csv", header=True, index=False)
    valid.to_csv(f"{dataset_dir}/{dataset}_valid.csv", header=True, index=False)
    test.to_csv(f"{dataset_dir}/{dataset}_test.csv", header=True, index=False)
