from transformers import AutoTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, BertConfig, BertTokenizer
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
import os

def test_vocab():
  model_names = ['bert-base-multilingual-cased', 'xlm-roberta-large', 'monologg/koelectra-base-v3-discriminator']

  for model_name in model_names:
    print(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab = tokenizer.get_vocab()
    print(len(vocab))
    # try:
    #   print(f"[SEP] : {vocab['[SEP]']}")
    # except:
    #   ...

    # try:
    #   print(f"</s> : {vocab['</s>']}")
    # except:
    #   ...

    # try:
    #   print(f"</e> : {vocab['</e>']}")
    # except:
    #   print("no </e>")

def test_tsv():
  train_df = pd.read_csv("/opt/ml/input/data/train/train.tsv")
  all_df = pd.read_csv("/opt/ml/input/data/train/all.tsv")

  
test_vocab()