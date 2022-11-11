# Benchmark for Readability Assessment in Spanish

Official repository for the paper "[A Benchmark for Neural Readability Assessment of Texts in Spanish](https://drive.google.com/file/d/1KdwvqrjX8MWYRDGBKeHmiR1NCzDcVizo/view?usp=share_link)"
by [@lmvasquezr](https://twitter.com/lmvasquezr), [@pcuenq](https://twitter.com/pcuenq), [@fireblend](https://twitter.com/fireblend) and [@feralvam](https://twitter.com/feralvam).

If you have any question, please don't hesitate to [contact us](mailto:lvasquezcr@gmail.com?subject=[GitHub]%20Investigating%20TS%20Eval%20Question). Feel free to submit any issue/enhancement in [GitHub](https://github.com/lmvasque/ts-explore/issues). 
## Datasets

Our datasets include a combination of texts that are freely available and with a data license agreement. Nevertheless, we have published the collected datasets that are freely available datasets in [HuggingFace](https://huggingface.co/) to support further readability studies. Please find the links in the table below:

|                                  Dataset                                  |                            Original Readability Level                            |
|:-------------------------------------------------------------------------:|:--------------------------------------------------------------------------------:|
|   [HablaCultura](https://huggingface.co/datasets/lmvasque/hablacultura)   | [CEFR](https://www.coe.int/en/web/common-european-framework-reference-languages) |
|         [kwiziq](https://huggingface.co/datasets/lmvasque/kwiziq)         | [CEFR](https://www.coe.int/en/web/common-european-framework-reference-languages) |
| [coh-metrix-esp](https://huggingface.co/datasets/lmvasque/coh-metrix-esp) |                                 simple, complex                                  |
|           [CAES](https://huggingface.co/datasets/lmvasque/caes)           | [CEFR](https://www.coe.int/en/web/common-european-framework-reference-languages) |
|   [Simplext*](https://sid-inico.usal.es/idocs/F8/FDO26144/Simplext.pdf)   |                                 simple, complex                                  |
|                   [Newsela*](https://newsela.com/data/)                   |             School Grade Levels (2-12) and Readability Levels (0-4)              |
| [OneStopCorpus](https://github.com/nishkalavallabhi/OneStopEnglishCorpus) |                          basic, intermediate, advanced                           |

*Please request your access for Newsela and Simplext corpus (Horacio Saggion) and we will be happy to share our splits upon request.


## Models

We have released all of our pretrained models in [HuggingFace](https://huggingface.co/):


| Model                                                                                                     | Granularity    | # classes |
|-----------------------------------------------------------------------------------------------------------|----------------|:---------:|
| [BERTIN (ES)](https://huggingface.co/lmvasque/readability-es-benchmark-bertin-es-paragraphs-2class)       | paragraphs     |     2     |
| [BERTIN (ES)](https://huggingface.co/lmvasque/readability-es-benchmark-bertin-es-paragraphs-3class)       | paragraphs     |     3     |
| [mBERT (ES)](https://huggingface.co/lmvasque/readability-es-benchmark-mbert-es-paragraphs-2class)         | paragraphs     |     2     |
| [mBERT (ES)](https://huggingface.co/lmvasque/readability-es-benchmark-mbert-es-paragraphs-3class)         | paragraphs     |     3     |
| [mBERT (EN+ES)](https://huggingface.co/lmvasque/readability-es-benchmark-mbert-en-es-paragraphs-3class) | paragraphs |     3     |
| [BERTIN (ES)](https://huggingface.co/lmvasque/readability-es-benchmark-bertin-es-sentences-2class)        | sentences      |     2     |
| [BERTIN (ES)](https://huggingface.co/lmvasque/readability-es-benchmark-bertin-es-sentences-3class)        | sentences      |     3     |
| [mBERT (ES)](https://huggingface.co/lmvasque/readability-es-benchmark-mbert-es-sentences-2class)          | sentences      |     2     |
| [mBERT (ES)](https://huggingface.co/lmvasque/readability-es-benchmark-mbert-es-sentences-3class)          | sentences      |     3     |
| [mBERT (EN+ES)](https://huggingface.co/lmvasque/readability-es-benchmark-mbert-en-es-sentences-3class) | sentences  |   3   |

For the zero-shot setting, we used the original models [BERTIN](bertin-project/bertin-roberta-base-spanish) and [mBERT](https://huggingface.co/bert-base-multilingual-uncased) with no further training. 
Also, you can find our ```TF-IDF+Logistic Regression``` approach in ```model_regression.py```, which is based on [this](https://www.kaggle.com/code/kashnitsky/logistic-regression-tf-idf-baseline/notebook) implementation.


## Reproducibility
We have published all of our datasets and models in [HuggingFace](https://huggingface.co/). However, as a reference, we have also included our training and data processing scripts in the [source folder](https://github.com/lmvasque/readability-es-benchmark/tree/main/src).

## Citation

If you use our results and scripts in your research, please cite our work: "[A Benchmark for Neural Readability Assessment of Texts in Spanish](https://drive.google.com/file/d/1KdwvqrjX8MWYRDGBKeHmiR1NCzDcVizo/view?usp=share_link)" (to be published) 

```
@inproceedings{vasquez-rodriguez-etal-2022-benchmarking,
    title = "A Benchmark for Neural Readability Assessment of Texts in Spanish",
    author = "V{\'a}squez-Rodr{\'\i}guez, Laura  and
      Cuenca-Jim{\'\e}nez, Pedro-Manuel and
      Morales-Esquivel, Sergio Esteban and
      Alva-Manchego, Fernando",
    booktitle = "Workshop on Text Simplification, Accessibility, and Readability (TSAR-2022), EMNLP 2022",
    month = dec,
    year = "2022",
}
```

### About the available datasets

We have downloaded the datasets below from its original website to make it available to the community in [HuggingFace](https://huggingface.co/). If you use this data, please credit the original author and our work as well.

#### CAES

We have extracted the CAES corpus from their [website](https://galvan.usc.es/caes/search). If you use this corpus, please also cite their work as follows:
```
@article{Parodi2015,
  author = "Giovanni Parodi",
  title = "Corpus de aprendices de español (CAES)",
  journal = "Journal of Spanish Language Teaching",
  volume = "2",
  number = "2",
  pages = "194-200",
  year  = "2015",
  publisher = "Routledge",
  doi = "10.1080/23247797.2015.1084685",
  URL = "https://doi.org/10.1080/23247797.2015.1084685",
  eprint = "https://doi.org/10.1080/23247797.2015.1084685"
}
```

#### Coh-Metrix-Esp (Cuentos)

We have made available in the HF the collected dataset from Coh-Metrix-Esp paper.  If you use their data, please cite their work as follows:
```
@inproceedings{quispesaravia-etal-2016-coh,
    title = "{C}oh-{M}etrix-{E}sp: A Complexity Analysis Tool for Documents Written in {S}panish",
    author = "Quispesaravia, Andre  and
      Perez, Walter  and
      Sobrevilla Cabezudo, Marco  and
      Alva-Manchego, Fernando",
    booktitle = "Proceedings of the Tenth International Conference on Language Resources and Evaluation ({LREC}'16)",
    month = may,
    year = "2016",
    address = "Portoro{\v{z}}, Slovenia",
    publisher = "European Language Resources Association (ELRA)",
    url = "https://aclanthology.org/L16-1745",
    pages = "4694--4698",
}
```

#### HablaCultura and Kwiziq

For these datasets, please also give credit to [HablaCultura.com](https://hablacultura.com/) and [Kwiziq](https://www.kwiziq.com/) websites. 