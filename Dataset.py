import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import pandas as pd


class TrainDataset(Dataset):
    def __init__(self, df, pretraine_path='aubmindlab/bert-base-arabert', max_length=128):
        self.df = df
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(pretraine_path)

    def __getitem__(self, index):
        text = self.df.iloc[index]['tweet']
        l_sarcasm = self.df.iloc[index]["sarcasm"]
        l_sentiment = self.df.iloc[index]["sentiment"]
        sentiment_dict = {
            "Positive": 2,
            "Neutral" : 1,
            "Negative": 0,
        }
        sarcasm_dict = {
            "True" : 1,
            "False" : 0
        }
        sentiment = sentiment_dict[l_sentiment]
        sarcasm = sarcasm_dict[str(l_sarcasm)]
        encoded_input = self.tokenizer(
                text,
                max_length = self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt",
            )

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None

        data_input = {
            "input_ids":input_ids.flatten(),
            "attention_mask": attention_mask.flatten()
        }

        label_input ={
            "sarcasm": torch.tensor(sarcasm, dtype=torch.float),
            "sentiment": torch.tensor(sentiment, dtype=torch.long),

        }

        return data_input, label_input

    def __len__(self):
        return self.df.shape[0]


class TestDataset(Dataset):
    def __init__(self, df, pretraine_path='aubmindlab/bert-base-arabert', max_length=128):
        self.df = df
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(pretraine_path)

    def __getitem__(self, index):
        text = self.df.iloc[index]["tweet"]

        encoded_input = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded_input["input_ids"]
        attention_mask = encoded_input["attention_mask"] if "attention_mask" in encoded_input else None

        data_input = {
            "input_ids": input_ids.flatten(),
            "attention_mask": attention_mask.flatten()
        }

        return data_input

    def __len__(self):
        return self.df.shape[0]
