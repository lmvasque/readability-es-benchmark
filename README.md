# Benchmark for Readability Assessment in Spanish

Official repository for the paper "[A Benchmark for Neural Readability Assessment of Texts in Spanish](https://drive.google.com/file/d/1KdwvqrjX8MWYRDGBKeHmiR1NCzDcVizo/view?usp=share_link)"
by [@lmvasquezr](https://twitter.com/lmvasquezr), [@pcuenq](https://twitter.com/pcuenq), [@fireblend](https://twitter.com/fireblend) and [@feralvam](https://twitter.com/feralvam).

If you have any question, please don't hesitate to [contact us](mailto:lvasquezcr@gmail.com?subject=[GitHub]%20Investigating%20TS%20Eval%20Question). Feel free to submit any issue/enhancement in [GitHub](https://github.com/lmvasque/ts-explore/issues). 
## Datasets

Our datasets include a combination of texts that are freely available and with a data license agreement. 
Please request your access to [Newsela](https://newsela.com/data/) and [Simplext](mailto:horacio.saggion@upf.edu) corpus and we will be happy to share our splits upon request.

Nevertheless, we have published the collected datasets that are freely available datasets in [HuggingFace](https://huggingface.co/) to support further readability studies. Please find the links in the table below:

|                                  Dataset                                  | Granularity  | Original Readability Level |
|:-------------------------------------------------------------------------:|:------------:|:--------------------------:|
|   [HablaCultura](https://huggingface.co/datasets/lmvasque/hablacultura)   |  Paragraphs  |          [CEFR](https://www.coe.int/en/web/common-european-framework-reference-languages)          |
|         [kwiziq](https://huggingface.co/datasets/lmvasque/kwiziq)         |   Paragraphs    |            [CEFR](https://www.coe.int/en/web/common-european-framework-reference-languages)            |
| [coh-metrix-esp](https://huggingface.co/datasets/lmvasque/coh-metrix-esp) |    Paragraphs     |       simple/complex       |
|           [CAES](https://huggingface.co/datasets/lmvasque/caes)           | Paragraphs |            [CEFR](https://www.coe.int/en/web/common-european-framework-reference-languages)            |

[//]: # ()
[//]: # (### Readability Levels Mapping)

[//]: # ()
[//]: # (Our classifers models were trained following with the following mapping of labels:)

[//]: # ()
[//]: # (|  Class   | Readability Level  |  Label  |)

[//]: # (|:--------:|:------------------:|:-------:|)

[//]: # (| 2-class  |       Simple       |    0    |)

[//]: # (| 2-class  |      Complex       |    1    |)

[//]: # (|   |         |      |)

[//]: # (| 3-class  |       Basic        |    0    |)

[//]: # (| 3-class  |    Intermediate    |    1    |)

[//]: # (| 3-class  |      Advanced      |    2    |)


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
| **[mBERT (EN+ES)](https://huggingface.co/lmvasque/readability-es-benchmark-mbert-en-es-sentences-3class)** | **sentences**  |   **3**   |

For the zero-shot setting, we used the original models [BERTIN](bertin-project/bertin-roberta-base-spanish) and [mBERT](https://huggingface.co/bert-base-multilingual-uncased) with no further training. 
Also, you can find our ```TF-IDF+Logistic Regression``` approach in ```model_regression.py```. This is based on [this](https://www.kaggle.com/code/kashnitsky/logistic-regression-tf-idf-baseline/notebook) implementation.


## Reproducibility
We have published all our datasets and models in [HuggingFace](https://huggingface.co/). However, as a reference, we have also included our training and data processing scripts in the [source folder](https://github.com/lmvasque/readability-es-benchmark/tree/main/src)

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