import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# 데이터 불러오기 (경로 수정)
df = pd.read_csv("label/labeled_data.csv")

label_map = {"Fear": 0, "Neutral": 1, "Greed": 2}
df["label"] = df["predicted_sentiment"].map(label_map)

train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["content"].tolist(), df["label"].tolist(), test_size=0.2, stratify=df["label"], random_state=42
)

# 모델 저장 경로 변경
model_save_path = "label/kcbert_model"

# 학습 후 저장
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)
print(f"✅ 모델 학습 완료! 모델이 {model_save_path} 에 저장됨.")
