import openai
import pandas as pd

openai.api_key = "YOUR_OPENAI_API_KEY"

def gpt_sentiment_analysis(text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "Classify this text as Fear, Neutral, or Greed."},
                  {"role": "user", "content": text}]
    )
    return response["choices"][0]["message"]["content"]

# 데이터 경로 변경
df = pd.read_csv("label/cleaned_data.csv")

df["predicted_sentiment"] = df["content"].apply(gpt_sentiment_analysis)

df.to_csv("label/labeled_data.csv", index=False)
print("✅ 자동 감성 레이블링 완료! labeled_data.csv 저장됨.")
