import pandas as pd
import pickle
import os
from load_data import *
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--keyword', type=str)
args = parser.parse_args()

with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)

train_dir = "/opt/ml/input/data/train/train.tsv"
train_pd = pd.read_csv(train_dir, delimiter='\t', header=None)

all_dir = "/opt/ml/input/data/train/all.tsv"
more_dir = "/opt/ml/input/data/train/more.tsv"
my_dir = "/opt/ml/input/data/my/my_test3.tsv"

more_pd = pd.read_csv(more_dir, delimiter='\t', header=None)
# more_pd = pd.read_csv(my_dir, delimiter='\t', header=None)

# keyword = args.keyword
keyword = "인물:직업/직함"
addition = more_pd[more_pd[8] == keyword]
print(f"before : {addition.shape[0]}")
addition = addition.drop_duplicates([1])
print(f"after : {addition.shape[0]}")
nums = [3, 4, 6, 7]

# 데이터 추가
count = 0
length = len(train_pd[train_pd[8] == keyword])
limit = 1500 - length
print(f"limit : {limit}")
with open(all_dir, 'a') as f:
    for line in addition.values:
        line = line.tolist()
        for i in nums:
            line[i] = str(line[i])
        temp = "\t".join(line)
        temp += '\n'
        f.write(temp)
        if count > limit:
            break
        count += 1
print(f"{keyword}, {count} data has added!")

    