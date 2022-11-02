import datasets
import transformers
from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

import utils

print(f"Transformers version: {transformers.__version__}")
print(f"Datasets version: {datasets.__version__}")


class ReadabilityMbert():

    def __init__(self, num_labels, dataset, model_id, init=True):
        self.base_model_id = model_id

        self.epoch = 10
        self.num_labels = num_labels
        self.learning_rate = 3e-6
        self.train_batch_size = 16
        self.eval_batch_size = 16
        self.save_strategy = "epoch"
        self.save_steps = 10

        self.output_data_dir = "./output"
        self.model_name = self.base_model_id.replace("-", "_")
        self.model_name = self.model_name.replace("/", "_")

        print(f"==== Model: {self.model_name} ====")
        self.model_dir = f"./model/{dataset}_{self.model_name}"

        if init:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.base_model_id,
                                                                            num_labels=self.num_labels)

            self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        self.train_dataset = ""
        self.valid_dataset = ""

    def compute_metrics(self, pred, stage="", title="", filename="", num_labels=""):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)

        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
        acc = accuracy_score(labels, preds)
        matrix = confusion_matrix(labels, preds, normalize=None)
        matrix_normalized = confusion_matrix(labels, preds, normalize='true')
        output = {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

        if "test" in stage:
            f1_3digits = round(f1, 3)
            filename = f"{filename}_{f1_3digits}".replace(" ", "_").lower()
            title = f"{title} {f1_3digits}"
            utils.plot_confusion(matrix, matrix_normalized, title, filename, num_labels)
            print(output)

        return output

    def tokenize(self, batch):
        text = [str(t) for t in batch["text"]]
        return self.tokenizer(text, padding="max_length", truncation=True)

    def prepare_data(self, data_files):
        dataset = load_dataset("csv", data_files=data_files)

        train_dataset = dataset["train"]
        valid_dataset = dataset["valid"]

        train_dataset = train_dataset.map(self.tokenize, batched=True, batch_size=len(train_dataset))
        valid_dataset = valid_dataset.map(self.tokenize, batched=True, batch_size=len(valid_dataset))

        self.train_dataset = train_dataset.remove_columns(["text", "source"])
        self.valid_dataset = valid_dataset.remove_columns(["text", "source"])

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.model_dir,
            num_train_epochs=self.epoch,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            save_strategy=self.save_strategy,
            save_steps=self.save_steps,
            evaluation_strategy=self.save_strategy,
            logging_strategy=self.save_strategy,
            load_best_model_at_end=True,
            learning_rate=self.learning_rate,
            save_total_limit=2,
            weight_decay=0.02
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            train_dataset=self.train_dataset,
            eval_dataset=self.valid_dataset
        )

        trainer.train()
        trainer.evaluate(eval_dataset=self.valid_dataset)
        trainer.save_model(self.model_dir + "_local")

    def predict(self, data_files, model_path, split, dataset, num_labels, tag=""):
        print(f"==== Predictions for: {split} ====")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=self.num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        test_dataset = load_dataset("csv", data_files=data_files, split=split)
        test_dataset = test_dataset.map(self.tokenize, batched=True, batch_size=len(test_dataset))

        test_trainer = Trainer(self.model)
        predictions = test_trainer.predict(test_dataset)

        if "para" in dataset:
            title = "Paragraph"
        if "sent" in dataset:
            title = "Sentence"
        title += f" ({num_labels}-class) - F1 score:"
        model_name = model_path.replace('./model/', '').replace('/', '_')
        filename = f"{tag}{model_name}"
        self.compute_metrics(predictions, "test", title, filename, num_labels)
