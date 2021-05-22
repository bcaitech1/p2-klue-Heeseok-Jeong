import pickle as pickle
import os
import pandas as pd
import torch

# Dataset 구성.
class RE_Dataset(torch.utils.data.Dataset):
  def __init__(self, tokenized_dataset, labels):
    self.tokenized_dataset = tokenized_dataset
    self.labels = labels

  def __getitem__(self, idx):
    item = {key: torch.tensor(val[idx]) for key, val in self.tokenized_dataset.items()}
    item['labels'] = torch.tensor(self.labels[idx])
    return item

  def __len__(self):
    return len(self.labels)

# 처음 불러온 tsv 파일을 원하는 형태의 DataFrame으로 변경 시켜줍니다.
# 변경한 DataFrame 형태는 baseline code description 이미지를 참고해주세요.
def preprocessing_dataset(dataset, label_type):
  labels = []
  for i, label in enumerate(dataset[8]):
    if label == 'blind':
      labels.append(100)
    else:
      labels.append(label_type[label])
  out_dataset = pd.DataFrame({'sentence':dataset[1],'entity_01':dataset[2],'entity_02':dataset[5],'label':labels,})
  return out_dataset

# tsv 파일을 불러옵니다.
def load_data(dataset_dir):
  # load label_type, classes
  with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)
  # load dataset
  dataset = pd.read_csv(dataset_dir, delimiter='\t', header=None)
  # preprecessing dataset
  dataset = preprocessing_dataset(dataset, label_type)
  
  return dataset

# bert input을 위한 tokenizing.
# tip! 다양한 종류의 tokenizer와 special token들을 활용하는 것으로도 새로운 시도를 해볼 수 있습니다.
# baseline code에서는 2가지 부분을 활용했습니다.
def tokenized_dataset(dataset, tokenizer, args):
  if args.input_style == 'qa1': # sen, e01 과 e02 는 무슨 관계일까요? 
    print("Using qa1 ")
    questions = []
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
      questions.append(e01 + ' 과 ' +  e02 + ' 는 무슨 관계일까요?')
    tokenized_sentences = tokenizer(
        list(dataset['sentence']),
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
        add_special_tokens=True,
  )
  elif args.input_style == 'qa2': # </e1> 이순신 </e1> 은 조선 중기 </e2> 무신 </e2> 이었다.
    print("Using qa2 ")
    sentences = []
    for sen, e01, e02 in zip(dataset['sentence'], dataset['entity_01'], dataset['entity_02']):
      sen = sen.replace(e01, f" </e1> {e01} </e1> ")
      sen = sen.replace(e02, f" </e2> {e02} </e2> ")
      sentences.append(sen)
    tokenized_sentences = tokenizer(
        sentences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
        add_special_tokens=True,
  )
  elif args.input_style == 'qa3': # </e1> 이순신 </e1> 은 조선 중기 </e2> 무신 </e2> 이었다. </e1> e01 </e1> 과 </e2> e02 </e2> 는 무슨 관계일까요? 
    print("Using qa3 ")
    sentences = []
    questions = []
    for sen, e01, e02 in zip(dataset['sentence'], dataset['entity_01'], dataset['entity_02']):
      sen = sen.replace(e01, f" </e1> {e01} </e1> ")
      sen = sen.replace(e02, f" </e2> {e02} </e2> ")
      sentences.append(sen)

      questions.append('</e01> ' + e01 + '</e01> 과 </e02> ' + e02 + ' </e02> 는 무슨 관계일까요?')
    tokenized_sentences = tokenizer(
        list(dataset['sentence']),
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
        add_special_tokens=True,
  )
  else: # base
    print("Using base ")
    concat_entity = []
    for e01, e02 in zip(dataset['entity_01'], dataset['entity_02']):
      sep = '</s>' if 'loberta' in args.pretrained_model else '[SEP]'
      temp = e01 + sep + e02
      concat_entity.append(temp)
    tokenized_sentences = tokenizer(
        concat_entity,
        list(dataset['sentence']),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_length,
        add_special_tokens=True,
    )
  return tokenized_sentences
