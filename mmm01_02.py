# mmm01_02.py

# 1. 필요한 라이브러리 임포트
import pandas as pd
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')  # 경고 메시지 무시 (필요 시 제거)

# 한국어 텍스트 처리를 위한 라이브러리 (kiwipiepy 사용)
from kiwipiepy import Kiwi
kiwi = Kiwi()

# 시각화를 위한 라이브러리
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 한글 폰트 설정 (Windows 환경에서 한글 깨짐 방지)
plt.rcParams['font.family'] = 'Malgun Gothic'  # Windows의 경우
# Mac의 경우: plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 2. 데이터 불러오기 (이전 단계에서 저장한 정제된 데이터 사용)
data_path = "cleaned_data.csv"
df = pd.read_csv(data_path)

# 컬럼 이름 확인
print("데이터의 컬럼:")
print(df.columns)

print("\n데이터의 첫 5행:")
print(df.head())

print(f"\n데이터 크기: {df.shape}")

# 3. 텍스트 정규화

# 3.1 형태소 분석 (토큰화)
def tokenize_text(text):
    # Kiwi를 사용한 형태소 분석
    tokens = kiwi.tokenize(text)
    # 명사, 동사, 형용사만 추출 (품사 태그: NNG, VV, VA 등)
    filtered_tokens = [token.form for token in tokens if token.tag in ['NNG', 'VV', 'VA']]
    return filtered_tokens

# 토큰화 적용
df['tokens'] = df['cleaned_content'].apply(tokenize_text)

# 토큰화 결과 확인
print("\n토큰화 결과 첫 5행:")
print(df[['cleaned_content', 'tokens']].head())

# 3.2 불용어 제거
# 간단한 불용어 사전 정의 (필요에 따라 확장 가능)
stopwords = ['이다', '하다', '되다', '같다', '없다', '있다', '보다', '받다', '가다', '오다']

def remove_stopwords(tokens):
    return [token for token in tokens if token not in stopwords]

# 불용어 제거 적용
df['filtered_tokens'] = df['tokens'].apply(remove_stopwords)

# 불용어 제거 결과 확인
print("\n불용어 제거 후 첫 5행:")
print(df[['tokens', 'filtered_tokens']].head())

# 4. 피처 엔지니어링

# 4.1 텍스트 길이 (이미 계산됨, 확인용 출력)
print("\n텍스트 길이 통계 (기존):")
print(df['text_length'].describe())

# 4.2 키워드 빈도 계산
# 모든 토큰을 하나의 리스트로 결합
all_tokens = sum(df['filtered_tokens'], [])

# 가장 빈도가 높은 키워드 10개 확인
token_counts = Counter(all_tokens)
top_keywords = token_counts.most_common(10)
print("\n가장 빈도가 높은 키워드 10개:")
print(top_keywords)

# 4.3 주식 관련 키워드 기반 감성 점수 계산
# 간단한 감성 사전 정의 (기획서에서 커스텀 감성 사전 구축 예정이므로 예시로 작성)
sentiment_dict = {
    '급등': 1, '상한가': 1, '대박': 1, '좋다': 1, '오르다': 1,  # 긍정 키워드
    '폭락': -1, '하락': -1, '나쁘다': -1, '망하다': -1, '떨어지다': -1  # 부정 키워드
}

def calculate_sentiment_score(tokens):
    score = 0
    for token in tokens:
        if token in sentiment_dict:
            score += sentiment_dict[token]
    return score

# 감성 점수 계산
df['sentiment_score'] = df['filtered_tokens'].apply(calculate_sentiment_score)

# 감성 점수 분포 확인
print("\n감성 점수 분포:")
print(df['sentiment_score'].describe())

# 5. 데이터 탐색 및 시각화

# 5.1 텍스트 길이 분포 시각화
plt.figure(figsize=(10, 6))
sns.histplot(df['text_length'], bins=50, kde=True)
plt.title('텍스트 길이 분포')
plt.xlabel('텍스트 길이')
plt.ylabel('빈도')
plt.show()

# 5.2 감성 점수 분포 시각화
plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment_score'], bins=20, kde=True)
plt.title('감성 점수 분포')
plt.xlabel('감성 점수')
plt.ylabel('빈도')
plt.show()

# 5.3 종목별 평균 감성 점수
stock_sentiment = df.groupby('stock_name')['sentiment_score'].mean().sort_values(ascending=False)
print("\n종목별 평균 감성 점수:")
print(stock_sentiment)

# 5.4 시간대별 감성 점수 변화
hourly_sentiment = df.groupby('hourly_timestamp')['sentiment_score'].mean()
plt.figure(figsize=(12, 6))
hourly_sentiment.plot()
plt.title('시간대별 평균 감성 점수 변화')
plt.xlabel('시간')
plt.ylabel('평균 감성 점수')
plt.xticks(rotation=45)
plt.show()

# 6. 데이터 저장
# 추가 피처가 포함된 데이터를 저장
final_data_path = "processed_data.csv"
df.to_csv(final_data_path, index=False, encoding='utf-8-sig')
print(f"\n처리된 데이터를 {final_data_path}에 저장했습니다.")
