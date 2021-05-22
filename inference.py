from transformers import AutoTokenizer, Trainer, TrainingArguments, BertTokenizer
from transformers import XLMRobertaForSequenceClassification, XLMRobertaConfig, BertForSequenceClassification, BertConfig
from torch.utils.data import DataLoader
from load_data import *
import pandas as pd
import torch
import pickle as pickle
import numpy as np
import argparse
import os

# import argparse
from importlib import import_module

def inference(model, tokenized_sent, device):
  dataloader = DataLoader(tokenized_sent, batch_size=40, shuffle=False)
  model.eval()
  logits = []
  output_pred = []
  
  for i, data in enumerate(dataloader):
    with torch.no_grad():
      if 'roberta' in args.pretrained_model:
        outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device)
        )
      else:
        outputs = model(
          input_ids=data['input_ids'].to(device),
          attention_mask=data['attention_mask'].to(device),
          token_type_ids=data['token_type_ids'].to(device)
        )
    _logits = outputs[0]
    _logits = _logits.detach().cpu().numpy()
    result = np.argmax(_logits, axis=-1)

    logits.append(_logits)
    output_pred.append(result)
  
  return np.concatenate(logits), np.array(output_pred).flatten()

def load_test_dataset(dataset_dir, tokenizer, args):
  test_dataset = load_data(dataset_dir)
  test_label = test_dataset['label'].values
  # tokenizing dataset
  tokenized_test = tokenized_dataset(test_dataset, tokenizer, args)

  return tokenized_test, test_label

def main(args):
  """
    주어진 dataset tsv 파일과 같은 형태일 경우 inference 가능한 코드입니다.
  """
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  # load tokenizer
  TOK_NAME = args.pretrained_model
  tokenizer = AutoTokenizer.from_pretrained(TOK_NAME)

  # load my model
  model_dir = args.model_dir # model dir.
  print(f"model_dir : {model_dir}")
  model_module = getattr(import_module("transformers"), args.model_type + "ForSequenceClassification")
  print(f"model_module : {model_module}")
  model = model_module.from_pretrained(model_dir)
  model.to(device)

  # load test datset
  test_dataset_dir = "/opt/ml/input/data/test/test.tsv"
  test_dataset, test_label = load_test_dataset(test_dataset_dir, tokenizer, args)
  test_dataset = RE_Dataset(test_dataset, test_label)

  # predict answer
  logits, pred_answer = inference(model, test_dataset, device)
  # make csv file with predicted answer
  # 아래 directory와 columns의 형태는 지켜주시기 바랍니다.

  output = pd.DataFrame(pred_answer, columns=['pred'])
  out_dir = './prediction/'
  os.makedirs(out_dir, exist_ok=True)
  out_file = model_dir.split('/')
  out_file = out_file[2] + '_' + out_file[3] + '.csv'
  out_path = out_dir + out_file
  print(f"prediction saved : {out_path}")
  output.to_csv(out_path, index=False)

  logits_path = out_dir + "logits/"
  os.makedirs(logits_path, exist_ok=True)
  logits_path += out_file.replace('.csv', '')
  np.save(logits_path, logits)
  print(f"logits saved : {logits_path}")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  
  # model dir
  parser.add_argument('--model_dir', type=str, default="./results/expr/checkpoint-500")
  parser.add_argument('--model_type', type=str, default="Bert")
  parser.add_argument('--pretrained_model', type=str, default="bert-base-multilingual-cased")
  parser.add_argument('--max_length', type=int, default=200)
  parser.add_argument('--input_style', type=str, default='base')
  args = parser.parse_args()
  print(args)
  main(args)
  
