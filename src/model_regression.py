from pathlib import Path

import nltk
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

import utils
from config import SEED

nltk.download('stopwords')


def train_regression(train_files, valid_files):
    train = pd.read_csv(train_files).dropna()
    train = train.drop(train[train.label == "N/A"].index)
    train = train.reset_index(drop=True)
    train.head()

    valid = pd.read_csv(valid_files).dropna()

    valid = valid.drop(valid[valid.label == "N/A"].index)
    valid = valid.reset_index(drop=True)
    valid.head()

    print(train.shape, valid.shape)

    stopword_es = nltk.corpus.stopwords.words('spanish')
    text_transformer = TfidfVectorizer(stop_words=stopword_es, ngram_range=(1, 2), lowercase=True, max_features=150000)

    x_train_text = text_transformer.fit_transform(train['text'])
    x_test_text = text_transformer.transform(valid['text'])

    x_train_text.shape, x_test_text.shape
    logit = LogisticRegression(C=5e1, solver='sag', multi_class='multinomial',
                               random_state=SEED, n_jobs=4, max_iter=200)
    logit.fit(x_train_text, train['label'])

    test_preds = logit.predict(x_test_text)

    labels = valid['label']
    preds = test_preds
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)

    print(" & ".join([str(round(s, 4)) for s in [acc, f1, precision, recall]]))

    matrix = confusion_matrix(labels, preds, normalize=None)
    matrix_normalized = confusion_matrix(labels, preds, normalize='true')

    f1_3digits = round(f1, 3)
    filename = Path(train_files).stem.replace("_train", "")

    if "para" in filename:
        title = "Paragraph"
    if "sent" in filename:
        title = "Sentence"

    filename = f"{filename}_lr_tf_idf_{f1_3digits}".replace(" ", "_").lower()

    num_labels = 3
    if "2" in filename:
        num_labels = 2

    filename = f"{filename}_lr_tf_idf_{f1_3digits}".replace(" ", "_").lower()
    title += f" ({num_labels}-class) - F1 score:"
    title = f"{title} {f1_3digits}"

    utils.plot_confusion(matrix, matrix_normalized, title, filename, num_labels)


def main():
    para3_class = "data/csv/para_3class/para_3class_{}.csv"
    para2_class = "data/csv/para_2class/para_2class_{}.csv"
    sent2_class = "data/csv/sent_2class/sent_2class_{}.csv"
    sent3_class = "data/csv/sent_3class/sent_3class_{}.csv"

    for t, name in [(para2_class, "para2"), (para3_class, "para3"), (sent2_class, "sent2"), (sent3_class, "sent3")]:
        print(name)
        train_regression(t.format("train"), t.format("test"))


main()
