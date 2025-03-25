import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 학습된 모델과 토크나이저 로드
model_path = "./kcbert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"사용 중인 디바이스: {device}")

# 감성 예측 함수 (공포/탐욕 반영)
def predict_sentiment(text):
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
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

# 테스트 데이터 준비
test_comments = [
    "삼성전자 대박이네요! 월요일 상한가 가자!",
    "이 종목 폭락했어요... 망했네요.",
    "주식이 오를 것 같아요. 좋네요.",
    "오늘 시장 흐름 보니까 별 변화 없을 듯.",
    "이건 그냥 지켜봐야 할 것 같네요."
]

# 예측 수행
print("\n테스트 댓글 감성 예측 결과:")
for comment in test_comments:
    sentiment = predict_sentiment(comment)
    print(f"댓글: {comment}")
    print(f"감성: {sentiment}\n")
