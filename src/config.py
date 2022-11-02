SEED = 345
DATA_DIR = "data/OneStopEnglishCorpus"

OSEC_READ_LEVELS = {
    "Ele-Txt": 0,
    "Int-Txt/Int-Txt": 1,
    "Int-Txt": 1,
    "Adv-Txt": 2,
}

DATA_DIRS = {
    "train": "data/csv/{0}/{0}_train.csv",
    "valid": "data/csv/{0}/{0}_valid.csv",
    "test": "data/csv/{0}/{0}_test.csv"
}


ROOT_READABILITY = "data/normalized"
PARAGRAPHS = [f"{ROOT_READABILITY}/coh/coh.cuentos.jsonl",
              f"{ROOT_READABILITY}/hablacultura/hablaculturajsonl",
              f"{ROOT_READABILITY}/kwiziq/kwiziq.jsonl",
              f"{ROOT_READABILITY}/newsela-es/newsela-es.paragraphs.jsonl",
              f"{ROOT_READABILITY}/simplext/simplext-part1-paragraphs.jsonl",
              f"{ROOT_READABILITY}/simplext/simplext-part2-paragraphs.jsonl"]

SENTENCES = [f"{ROOT_READABILITY}/newsela-es/newsela-es.sentences.jsonl",
             f"{ROOT_READABILITY}/simplext/simplext-part1-sentences.jsonl",
             f"{ROOT_READABILITY}/simplext/simplext-part2-sentences.jsonl"]