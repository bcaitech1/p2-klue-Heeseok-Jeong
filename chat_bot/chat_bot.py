import torch
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import argparse
import os

def get_model(MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    return tokenizer, model

def get_cls_token(sent_A):
    model.eval()
    tokenized_sent = tokenizer(
            sent_A,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=128
    )
    with torch.no_grad():# 그라디엔트 계산 비활성화
        outputs = model(    # **tokenized_sent
            input_ids=tokenized_sent['input_ids'],
            attention_mask=tokenized_sent['attention_mask'],
            token_type_ids=tokenized_sent['token_type_ids']
            )
    logits = outputs.last_hidden_state[:,0,:].detach().cpu().numpy()
    return logits

def feed_dataset():
    chatbot_Question = ['기차 타고 여행 가고 싶어','꿈이 이루어질까?','내년에는 더 행복해질려고 이렇게 힘든가봅니다', '간만에 휴식 중', '오늘도 힘차게!'] # 질문
    chatbot_Answer = ['꿈꾸던 여행이네요.','현실을 꿈처럼 만들어봐요.','더 행복해질 거예요.', '휴식도 필요하죠', '아자아자 화이팅!!'] # 답변

    data_dir = '/opt/ml/code/chat_bot/ChatbotData.csv'
    df = pd.read_csv(data_dir)

    additional_question = df['Q']
    additional_answer = df['A']
    chatbot_Question += additional_question.tolist()
    chatbot_Answer += additional_answer.tolist()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Heath\'s ChatBot ><')
    parser.add_argument('--train', type=int, default=0, help='set train mode')
    
    args = parser.parse_args()

    if args.train:
        model_dir = '/opt/ml/code/chat_bot/ChatbotData.csv'
        out_dir = './results'
        os.makedirs(out_dir, exist_ok=True)
        train_model(model_dir, out_dir)
        
        