# train_model.py

# 1. 필요한 라이브러리 임포트
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import warnings
warnings.filterwarnings('ignore')

# 2. GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# 3. 데이터 불러오기
data_path = "processed_data.csv"
df = pd.read_csv(data_path)

# 컬럼 이름 확인
print("데이터의 컬럼:")
print(df.columns)

print("\n데이터의 첫 5행:")
print(df.head())

print(f"\n데이터 크기: {df.shape}")

# 4. 감성 레이블 준비
# sentiment_score를 기반으로 이진 분류 레이블 생성 (0: 부정, 1: 긍정)
df['label'] = df['sentiment_score'].apply(lambda x: 1 if x >= 0 else 0)

# 레이블 분포 확인
print("\n레이블 분포:")
print(df['label'].value_counts())

# 5. 데이터셋 준비
# KcBERT 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")

# 텍스트와 레이블 추출
texts = df['cleaned_content'].tolist()
labels = df['label'].tolist()

# 데이터셋을 학습/테스트로 분리
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

# 토크나이징 함수 정의
def tokenize_function(texts):
    return tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

# 학습 및 테스트 데이터 토크나이징
train_encodings = tokenize_function(train_texts)
test_encodings = tokenize_function(test_texts)

# PyTorch Dataset 클래스 정의
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# 학습 및 테스트 데이터셋 생성
train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

# 6. KcBERT 모델 로드 및 디바이스 설정
model = AutoModelForSequenceClassification.from_pretrained(
    "beomi/kcbert-base",
    num_labels=2  # 이진 분류 (긍정/부정)
)
model.to(device)  # 모델을 GPU 또는 CPU로 이동

# 7. 학습 설정
training_args = TrainingArguments(
    output_dir="./kcbert_results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 평가 메트릭 정의
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 8. 모델 학습
trainer.train()

# 9. 모델 평가
eval_results = trainer.evaluate()
print("\n모델 평가 결과:")
print(eval_results)

# 10. 모델 및 토크나이저 저장
model.save_pretrained("./kcbert_model")
tokenizer.save_pretrained("./kcbert_model")
print("\n모델과 토크나이저를 ./kcbert_model에 저장했습니다.")