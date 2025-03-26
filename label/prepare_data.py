import pandas as pd

# 데이터 불러오기
df = pd.read_csv("label/processed_data.csv")  # 폴더 경로 추가

# 필요 없는 컬럼 제거
df = df[["content"]]

# 저장 경로 변경
df.to_csv("label/cleaned_data.csv", index=False)
print("✅ 데이터 전처리 완료! cleaned_data.csv 저장됨.")
