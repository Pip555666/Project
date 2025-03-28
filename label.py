from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import pandas as pd

# KcBERT 모델 로드
model_name = "beomi/KcBERT-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)  # 긍정/부정 2개 클래스
model.eval()

# 감성 분석 함수 (공포/탐욕 추가 반영)
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_score = probs[0][1].item()  # 긍정 확률을 감정 점수로 활용
    
    if sentiment_score > 0.7:
        return "탐욕"
    elif sentiment_score < 0.3:
        return "공포"
    else:
        return "중립"

# 1️⃣ 크롤링한 CSV 파일 불러오기
input_file = "cleaned_data.csv"  # 크롤링한 데이터 파일 경로
df = pd.read_csv(input_file, encoding="utf-8-sig")

# 2️⃣ 감성 분석 수행
df["감성"] = df["content"].astype(str).apply(predict_sentiment)  # "content" 컬럼이 댓글 내용

# 3️⃣ 감성 분석이 포함된 새로운 CSV 파일 저장
output_file = "crawled_data_with_sentiment.csv"
df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"감성 분석 완료! 결과 파일: {output_file}")
