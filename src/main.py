import glob
import random

import numpy as np
import torch

from config import DATA_DIRS, SEED
from model_mbert import ReadabilityMbert

np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)

mappings = {
    "para_2class": 2,
    "para_3class": 3,
    "sent_2class": 2,
    "sent_3class": 3
}


def prepare_data_files(dataset):
    data_files = DATA_DIRS.copy()
    for file in DATA_DIRS:
        data_files[file] = DATA_DIRS[file].format(dataset)

    return data_files


def predict_models(num_labels, dataset, model_id):
    data_files = prepare_data_files(dataset)
    model = ReadabilityMbert(num_labels, dataset, model_id, init=False)
    checkpoints = glob.glob(f"{model.model_dir}/check*")
    checkpoints.append(f"{model.model_dir}_local")

    for ckp in checkpoints:
        print(f"==== Predicting for model: {ckp} ====")
        model.predict(data_files, ckp, "test", dataset, num_labels)


def train_multi_in_en():
    for ckp in ["./model/osec_bert_base_multilingual_uncased/checkpoint-203",
                "./model/osec_bert_base_multilingual_uncased/checkpoint-290",
                "./model/osec_bert_base_multilingual_uncased_local"]:
        for dataset, num_labels in mappings.items():
            print(dataset)
            print(f"==== Running readability training for: {dataset} ====")
            data_files = prepare_data_files(dataset)
            model = ReadabilityMbert(num_labels, dataset, model_id=ckp, init=False)
            model.predict(data_files, ckp, "test", dataset, num_labels, f"multi_en_zero_{dataset}")


def train_fine_tuning(model_id):
    for dataset, num_labels in mappings.items():
        print(f"==== Running readability training for: {dataset} ====")
        model = ReadabilityMbert(num_labels, dataset, model_id)
        data_files = prepare_data_files(dataset)
        model.prepare_data(data_files)
        model.train()
        predict_models(num_labels, dataset, model_id)


def run_zero_shot(model_id):
    for dataset, num_labels in mappings.items():
        print(dataset)
        data_files = prepare_data_files(dataset)
        model = ReadabilityMbert(num_labels, dataset, model_id=model_id, init=False)
        model.predict(data_files, model_id, "test", dataset, num_labels, f"zero_{dataset}_")


def run_mbert(model_id="bert-base-multilingual-uncased"):
    train_fine_tuning(model_id)
    train_multi_in_en()
    run_zero_shot(model_id)


def run_bertin(model_id="bertin-project/bertin-roberta-base-spanish"):
    train_fine_tuning(model_id)
    run_zero_shot(model_id)


def main():
    run_mbert()
    run_bertin()


if __name__ == '__main__':
    main()
