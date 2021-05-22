import pandas as pd
import pickle
import os
from load_data import *

with open('/opt/ml/input/data/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)

train_dir = "/opt/ml/input/data/train/train.tsv"
train_pd = pd.read_csv(train_dir, delimiter='\t', header=None)

test_dir = "/opt/ml/input/data/test/test.tsv"
test_pd = pd.read_csv(test_dir, delimiter='\t', header=None)

more_dir = "/opt/ml/input/data/train/more.tsv"
more_pd = pd.read_csv(more_dir, delimiter='\t', header=None)

all_dir = "/opt/ml/input/data/train/all.tsv"
# all_dir = "/opt/ml/input/data/my/my_test3.tsv"
# all_pd = pd.read_csv(all_dir, delimiter='\t', header=None, error_bad_lines=False)
all_pd = pd.read_csv(all_dir, delimiter='\t', header=None)


# train, test 데이터 레이블 개수 체크
print(train_pd[8].value_counts())
print(train_pd.shape)
print(train_pd[8].shape[0])
print()
# print(test_pd[8].value_counts())
# -> train 데이터 불균형 심함, test 는 모두 blind 처리

# more 데이터 레이블 개수 체크
print(more_pd[8].value_counts())
print(more_pd.shape)
print(more_pd[8].shape[0])
print()

# all 데이터 레이블 개수 체크
print(all_pd[8].value_counts())
print(all_pd.shape)
print(all_pd[8].shape[0])
print()


# more 각종 유니크 확인
# print(more_pd[1].unique().shape)
# print()

# 레이블 대분류 체크
# train_label = train_pd[8].value_counts().tolist()
# print(train_label)

# 955번째 행의 피쳐들 보기
# print(train_pd.iloc[955])
# print()

# submission 분석
# result_dir = "/opt/ml/code/prediction/submission1.csv"
# result_pd = pd.read_csv(result_dir, delimiter='\t', header=None)
# print(result_pd[0].value_counts())
# print(result_pd.info())
# print()





