# Heath's KLUE Relation Extraction


## [목차]

* [\[Relation Extraction 소개\]](#relation-extraction-소개)
* [\[Data\]](#data)
* [\[Installation\]](#installation)
    * [Dependencies](#dependencies)
* [\[Usage\]](#usage)
* [\[File Structure\]](#file-structure)
* [\[My Solution\]](#my-solution)
* [\[Reference\]](#reference)
    * [Library](#library)
    * [Papers](#papers)
    * [Dataset](#dataset)


<br>
<br>

## [Relation Extraction 소개]

관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122560946-30e88900-d07c-11eb-9b43-33c49a813ca9.png' height='250px '/>
</div>
<br/>

위 그림의 예시와 같이 요약된 정보를 사용해 QA 시스템 구축과 활용이 가능하며, 이외에도 요약된 언어 정보를 바탕으로 효율적인 시스템 및 서비스 구성이 가능합니다.

```
input: sentence, entity1, entity2 의 정보를 입력으로 사용 합니다.

sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
entity 1: 썬 마이크로시스템즈
entity 2: 오라클

relation: 단체:별칭
```

위 예시문에서 단체:별칭의 label은 6번(아래 label_type.pkl 참고)이며, 즉 모델이 sentence, entity 1과 entity 2의 정보를 사용해 label 6을 맞추는 분류 문제입니다.

`output : relation 42개 classes 중 1개의 class`

`평가 지표 : Accuracy`

## [Data]
학습을 위한 데이터는 총 9000개 이며, 1000개의 test 데이터가 있습니다.

label_type.pkl: 총 42개 classes pickle로 load하게 되면, 딕셔너리 형태의 정보를 얻을 수 있습니다.

```
with open('./dataset/label_type.pkl', 'rb') as f:
    label_type = pickle.load(f)

{'관계_없음': 0, '인물:배우자': 1, '인물:직업/직함': 2, '단체:모회사': 3, '인물:소속단체': 4, '인물:동료': 5, '단체:별칭': 6, '인물:출신성분/국적': 7, '인물:부모님': 8, '단체:본사_국가': 9, '단체:구성원': 10, '인물:기타_친족': 11, '단체:창립자': 12, '단체:주주': 13, '인물:사망_일시': 14, '단체:상위_단체': 15, '단체:본사_주(도)': 16, '단체:제작': 17, '인물:사망_원인': 18, '인물:출생_도시': 19, '단체:본사_도시': 20, '인물:자녀': 21, '인물:제작': 22, '단체:하위_단체': 23, '인물:별칭': 24, '인물:형제/자매/남매': 25, '인물:출생_국가': 26, '인물:출생_일시': 27, '단체:구성원_수': 28, '단체:자회사': 29, '인물:거주_주(도)': 30, '단체:해산일': 31, '인물:거주_도시': 32, '단체:창립일': 33, '인물:종교': 34, '인물:거주_국가': 35, '인물:용의자': 36, '인물:사망_도시': 37, '단체:정치/종교성향': 38, '인물:학교': 39, '인물:사망_국가': 40, '인물:나이': 41} 
```

<div align='center'>
    <img src='https://user-images.githubusercontent.com/37643891/122561391-c2f09180-d07c-11eb-8959-5829527b0aef.png' height='250px '/>
</div>
<br/>

```
column 1: 데이터가 수집된 정보.
column 2: sentence.
column 3: entity 1
column 4: entity 1의 시작 지점.
column 5: entity 1의 끝 지점.
column 6: entity 2
column 7: entity 2의 시작 지점.
column 8: entity 2의 끝 지점.
column 9: entity 1과 entity 2의 관계를 나타내며, 총 42개의 classes가 존재함.
```

<br>
<hr>
<br>

## [Installation]


### Dependencies

- torch==1.6.0
- pandas==1.1.5
- scikit-learn~=0.24.1
- transformers==4.2.0

```python
pip install -r requirements.txt
```

<br>
<br>

## [Usage]

### Train

모델을 학습하기 위해서는 `train.py` 를 실행시킵니다.

```bash
$ p2-klue-Heeseok-Jeong# python train.py
```

```
# train arguments
parser.add_argument('--model_type', type=str, default='Bert')
parser.add_argument('--pretrained_model', type=str, default='bert-base-multilingual-cased')
parser.add_argument('--seed' , type=int , default = 1331)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--output_dir', type=str, default='./results/expr')
parser.add_argument('--save_total_limit', type=int, default=3)
parser.add_argument('--max_length', type=int, default=200)
``` 

모델은 `Huggingface` 가 제공하는 모델을 사용할 수 있습니다.

- **Bert**
- **XLM_Roberta**
- **KoElectra**
...


### Inference

학습된 모델로 추론하기 위해서는 `inference.py` 를 실행시킵니다.

```bash
$ p4-dkt-no_caffeine_no_gain# python inference.py --model_name "학습한 모델 폴더 이름" --model_epoch "사용하고픈 모델의 epoch"
```

```
# valid arguments
parser.add_argument('--model_dir', type=str, default="./results/expr/checkpoint-500")
parser.add_argument('--model_type', type=str, default="Bert")
parser.add_argument('--pretrained_model', type=str, default="bert-base-multilingual-cased")
parser.add_argument('--max_length', type=int, default=200)
parser.add_argument('--input_style', type=str, default='base')
```

<br>
<br>

## [File Structure]

```python
p2-klue-Heeseok-Jeong/
│
├── EDA.ipynb
├── EDA.py
├── README.md
├── data_combine.py
├── ensemble.py
├── requirements.txt
├── inference.py
├── evaluation.py
├── trian.py
└── data_utils.py
```

<br>
<br>

## [My Solution]
- [Wrap Up Report](https://www.notion.so/Wrap-Up-P-Stage2-KLUE-_T1194-6210254e1ab04583b969993377e0567)

1. BERT, XLM_Roberta_large, monologg/koelectra 모델 사용
2. 문장 입력구조 변경
  - [CLS] 이순신 [SEP] 무신 [SEP] 이순신은 조선 중기 무신이다. [SEP]
  - [CLS] 이순신은 조선 중기 무신이다. [SEP] 이순신 과 무신 은 무슨 관계일까요? [SEP]
  - [CLS] </e1> 이순신 </e1> 은 조선 중기 </e2> 무신 </e2> 이다. [SEP]
  - [CLS] </e1> 이순신 </e1> 은 조선 중기 </e2> 무신 </e2> 이다. [SEP] </e1> 이순신 </e1> 과 </e2> 무신 </e2> 은 무슨 관계일까요? [SEP]
3. Ensemble - Soft Voting

<br>
<br>

## [Reference]

### Library
- [Huggingface](https://huggingface.co/)
- [Huggingface.transformers](https://huggingface.co/transformers/)



### Papers

- [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding (Devlin et al., arXiv 2018)](https://arxiv.org/pdf/1810.04805.pdf)
- [RoBERTa: A Robustly Optimized BERT Pretraining Approach (Liu et al., arXiv 2019)](https://arxiv.org/pdf/1907.11692.pdf)
- [Unsupervised Cross-lingual Representation Learning at Scale (Conneau et al., arXiv 2020)](https://arxiv.org/pdf/1911.02116.pdf)
- [ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators (Clark et al., arXiv 2020)](https://arxiv.org/pdf/2003.10555.pdf)


### Dataset

- [KLUE](https://github.com/KLUE-benchmark/KLUE)
