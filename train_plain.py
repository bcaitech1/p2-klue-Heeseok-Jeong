import pickle as pickle
import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from transformers import AutoTokenizer, AutoModel, AutoConfig, Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import XLMRobertaForSequenceClassification, XLMRobertaConfig, BertForSequenceClassification, BertConfig, ElectraForSequenceClassification, ElectraConfig
from load_data import *

import argparse
from importlib import import_module
from pathlib import Path
import glob
import re
import numpy as np
import random

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def increment_output_dir(output_path, exist_ok=False):
  path = Path(output_path)
  if (path.exists() and exist_ok) or (not path.exists()):
    return str(path)
  else:
    dirs = glob.glob(f"{path}*")
    matches = [re.search(rf"%s(\d+)" %path.stem, d) for d in dirs]
    i = [int(m.groups()[0]) for m in matches if m]
    n = max(i) + 1 if i else 2
    return f"{path}{n}"

# 평가를 위한 metrics function.
def compute_metrics(pred):
  labels = pred.label_ids
  preds = pred.predictions.argmax(-1)
  # calculate accuracy using sklearn's function
  acc = accuracy_score(labels, preds)
  return {
      'accuracy': acc,
  }

def train(args):
  seed_everything(args.seed)

  # load dataset
  # whole_dataset = load_data("/opt/ml/input/data/train/train.tsv")
  whole_dataset = load_data("/opt/ml/input/data/train/all.tsv")
  whole_label = whole_dataset['label'].values

  train_dataset, val_dataset= train_test_split(whole_dataset, test_size=0.1, random_state=args.seed)

  # load model and tokenizer
  MODEL_NAME = args.pretrained_model
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
  special_tokens = ['</e1>', '</e2>']
  special_tokens_dct = {'additional_special_tokens': special_tokens}
  tokenizer.add_special_tokens(special_tokens_dct)
    
  # tokenizing dataset
  tokenized_train = tokenized_dataset(train_dataset, tokenizer, args)
  tokenized_val = tokenized_dataset(val_dataset, tokenizer, args)

  # make dataset for pytorch.
  RE_train_dataset = RE_Dataset(tokenized_train, train_dataset['label'].values)
  RE_val_dataset = RE_Dataset(tokenized_val, val_dataset['label'].values)

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

  # setting model hyperparameter
  config_module = getattr(import_module("transformers"), args.model_type + "Config")
  model_config = config_module.from_pretrained(MODEL_NAME)
  # model_config = AutoConfig.from_pretrained(MODEL_NAME)
  model_config.num_labels = 42

  model_module = getattr(import_module("transformers"), args.model_type + "ForSequenceClassification")
  model = model_module.from_pretrained(MODEL_NAME, config=model_config)
  # model = AutoModel.from_pretrained(MODEL_NAME, config=model_config) 
  model.resize_token_embeddings(len(tokenizer))
  model.to(device)

  output_dir = increment_output_dir(args.output_dir)
  print(f"output_dir : {output_dir}")
  
  # 사용한 option 외에도 다양한 option들이 있습니다.
  # https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments 참고해주세요.
  training_args = TrainingArguments(
    output_dir=output_dir,          # output directory
    save_total_limit=args.save_total_limit,              # number of total save model.
    save_steps=500,                 # model saving step.
    num_train_epochs=args.epochs,              # total number of training epochs
    # save_strategy='epoch',                # also save at last time
    fp16 = True,
    dataloader_num_workers=4,
    label_smoothing_factor=0.5,
    learning_rate=args.lr,               # learning_rate
    per_device_train_batch_size=args.batch_size,  # batch size per device during training
    #per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=300,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,              # log saving step.
    evaluation_strategy='steps', # evaluation strategy to adopt during training
                                # `no`: No evaluation during training.
                                # `steps`: Evaluate every `eval_steps`.
                                # `epoch`: Evaluate every end of epoch.
    load_best_model_at_end=True,
    metric_for_best_model='loss',
    eval_steps = 500,            # evaluation step.
  )

  early_stopping = EarlyStoppingCallback(early_stopping_patience = 5, early_stopping_threshold = 0.001)

  trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    callbacks=[early_stopping],
    train_dataset=RE_train_dataset,         # training dataset
    eval_dataset=RE_val_dataset,             # evaluation dataset
    compute_metrics=compute_metrics         # define metrics function
  )

  # train model
  trainer.train()

  # save last model
  # trainer.save_model(output_dir)
  # trainer.save_state()

def main(args):
  train(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='Bert')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-multilingual-cased')
    parser.add_argument('--seed' , type=int , default = 1331)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--output_dir', type=str, default='./results/expr')
    parser.add_argument('--save_total_limit', type=int, default=3)
    parser.add_argument('--max_length', type=int, default=200)
    parser.add_argument('--input_style', type=str, default='base')
    

    args = parser.parse_args()
    
    main(args)
