# Deep Multi-Task Model for Sarcasm Detection and Sentiment Analysis in Arabic Language

## Requirement
1. pytorch
2. transformers
3. scikitlearn
4. pandas
5. barbar

## Model Training 

```
python train_model.py [args]
```
Options:
- lm_pretrained : pretrained BERT model (MARBERT, ARBERT, AraBERT, ...)
- lr : learning rate
- batch_size : batch size
- epochs : number of epochs
- lr_mult : Classifier learning rate multiplier

## Model evaluation/testing
```
python eval_model.py --lm_pretrained [value] --batch_size [value]
```


Citing this work
---------------------

If you are using this source code please use the following citation to reference this work:

```
@inproceedings{el-mahdaouy-etal-2021-deep,
    title = "Deep Multi-Task Model for Sarcasm Detection and Sentiment Analysis in {A}rabic Language",
    author = "El Mahdaouy, Abdelkader  and
      El Mekki, Abdellah  and
      Essefar, Kabil  and
      El Mamoun, Nabil  and
      Berrada, Ismail  and
      Khoumsi, Ahmed",
    booktitle = "Proceedings of the Sixth Arabic Natural Language Processing Workshop",
    month = apr,
    year = "2021",
    address = "Kyiv, Ukraine (Virtual)",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/2021.wanlp-1.42",
    pages = "334--339",
}	
```