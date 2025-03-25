import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import warnings
warnings.filterwarnings('ignore')

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# 데이터 불러오기
data_path = "processed_data.csv"
df = pd.read_csv(data_path)

# 데이터 확인
print("데이터의 컬럼:")
print(df.columns)
print("\n데이터의 첫 5행:")
print(df.head())
print(f"\n데이터 크기: {df.shape}")

# 감성 레이블 준비 (3분류: 공포=0, 중립=1, 탐욕=2)
df['label'] = df['sentiment_score'].apply(lambda x: 2 if x > 0.7 else (0 if x < 0.3 else 1))

# 레이블 분포 확인
print("\n레이블 분포:")
print(df['label'].value_counts())

# 데이터셋 준비
tokenizer = AutoTokenizer.from_pretrained("beomi/kcbert-base")
texts = df['cleaned_content'].tolist()
labels = df['label'].tolist()

# 데이터셋 분리
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, labels, test_size=0.2, random_state=42, stratify=labels
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

# 토크나이징 진행
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

# 데이터셋 생성
train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

# 모델 로드 (3분류)
model = AutoModelForSequenceClassification.from_pretrained(
    "beomi/kcbert-base", num_labels=3
)
model.to(device)

# 학습 설정
training_args = TrainingArguments(
    output_dir="./kcbert_results",
    num_train_epochs=5,  # 학습 epoch 증가
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,  # 더 자주 로그 출력
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# 평가 메트릭 정의
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# 모델 학습
trainer.train()

# 모델 평가
eval_results = trainer.evaluate()
print("\n모델 평가 결과:")
print(eval_results)

# 모델 저장
model.save_pretrained("./kcbert_model")
tokenizer.save_pretrained("./kcbert_model")
print("\n모델과 토크나이저를 ./kcbert_model에 저장했습니다.")
