import glob
import json
from pathlib import Path

import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from config import *


def prepare_one_stop_data():
    data_path = f"{DATA_DIR}/Texts-SeparatedByReadingLevel/"
    categories = ["Ele-Txt", "Adv-Txt", "Int-Txt/Int-Txt"]
    output_file = f"{DATA_DIR}/osec.jsonl"
    process_data(data_path, categories, OSEC_READ_LEVELS, output_file)


def process_data(data_path, categories, labels_map, output_file):
    for cat in categories:
        path = data_path + cat

        for f in glob.glob(f"{path}/*.txt"):
            with open(f, "r", encoding="utf-8", errors='ignore') as f1, open(output_file, "a+") as outfile:
                article_name = Path(f).name
                label = labels_map[cat]
                text = [line.strip() for line in f1.readlines()]
                entry = {"text": "".join(text), "label": label, "source": article_name}
                json.dump(entry, outfile)
                outfile.write("\n")


def plot_confusion(matrix, matrix_normalized, title, filename, num_labels):
    print(matrix)
    sns.set(font_scale=2)
    plt.figure(figsize=(9, 9))

    if num_labels == 3:
        labels = ["basic", "intermediate", "advanced"]
    elif num_labels == 2:
        labels = ["simple", "complex"]

    annotations = [f"{total}\n{percent:.2%}" for total, percent in
                   zip(matrix.flatten(), matrix_normalized.flatten())]
    annotations = np.asarray(annotations).reshape(matrix_normalized.shape)

    sns.heatmap(matrix, annot=annotations, cmap='Blues_r',  # annot_kws={'size': 16},
                fmt="", linewidths=.5,
                square=True,
                xticklabels=labels,
                yticklabels=labels)

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(title, size=22)
    plt.savefig(f"img/{filename}.png")
