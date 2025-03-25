# 1. 필요한 라이브러리 임포트
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 2. 학습된 모델과 토크나이저 로드
model_path = "./kcbert_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
model.eval()

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"사용 중인 디바이스: {device}")

# 3. 예측 함수 정의
def predict_sentiment(text):
    # 텍스트 토크나이징
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # 입력 데이터를 디바이스로 이동
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # 예측
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()

    # 감성 레이블 매핑 (0: 부정, 1: 중립, 2: 긍정)
    sentiment_labels = {0: "부정", 1: "중립", 2: "긍정"}
    return sentiment_labels.get(prediction, "알 수 없음")

# 4. 테스트 데이터 준비
test_comments = [
    "삼성전자 대박이네요! 월요일 상한가 가자!",
    "이 종목 폭락했어요... 망했네요.",
    "주식이 오를 것 같아요. 좋네요.",
    "오늘 시장 흐름 보니까 별 변화 없을 듯.",
    "이건 그냥 지켜봐야 할 것 같네요."
]

# 5. 예측 수행
print("\n테스트 댓글 감성 예측 결과:")
for comment in test_comments:
    sentiment = predict_sentiment(comment)
    print(f"댓글: {comment}")
    print(f"감성: {sentiment}\n")
