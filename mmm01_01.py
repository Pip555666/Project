# mmm01_01.py

# 1. 필요한 라이브러리 임포트
import pandas as pd
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')  # 경고 메시지 무시 (필요 시 제거)

# 한국어 텍스트 처리를 위한 라이브러리 (kiwipiepy 사용)
from kiwipiepy import Kiwi
kiwi = Kiwi()

# 2. 데이터 불러오기
data_path = "toss_삼성전자.csv"
df = pd.read_csv(data_path)

# 컬럼 이름 확인
print("데이터의 컬럼:")
print(df.columns)

print("\n데이터의 첫 5행:")
print(df.head())

print("\n데이터 기본 정보:")
print(df.info())

print(f"\n데이터 크기: {df.shape}")

# 3. 데이터 구조 파악 및 초기 정제

# 3.1 결측값 처리
print("\n결측값 개수:")
print(df.isnull().sum())

# content 컬럼에 결측값이 있는 행 제거
df = df.dropna(subset=['content'])
df = df[df['content'].str.strip() != '']  # 공백만 있는 댓글 제거

print(f"\n결측값 제거 후 데이터 크기: {df.shape}")

# 3.2 중복 데이터 제거
# 동일한 content와 timestamp 기준으로 중복 제거
duplicates = df.duplicated(subset=['content', 'timestamp']).sum()
print(f"\n중복 데이터 개수: {duplicates}")

df = df.drop_duplicates(subset=['content', 'timestamp'], keep='first')

print(f"중복 제거 후 데이터 크기: {df.shape}")

# 3.3 기본 텍스트 정제
def clean_text(text):
    # URL 제거
    text = re.sub(r'(?:http|https|www)\S+', '', text, flags=re.MULTILINE)
    # \r\n 같은 개행문자 제거
    text = text.replace('\r\n', ' ')
    # 특수문자 제거 (한글, 영어, 숫자, 공백 제외)
    text = re.sub(r'[^가-힣a-zA-Z0-9\s]', '', text)
    # 반복된 공백 제거
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 텍스트 정제 적용
df['cleaned_content'] = df['content'].apply(clean_text)

# 정제된 텍스트 확인
print("\n정제된 텍스트 첫 5행:")
print(df[['content', 'cleaned_content']].head())

# 3.4 URL만 포함된 댓글 제거
# 정제 후 텍스트가 비어 있는 행 제거
df = df[df['cleaned_content'].str.strip() != '']
print(f"\n빈 텍스트 제거 후 데이터 크기: {df.shape}")

# 3.5 시간 데이터 변환
# timestamp를 datetime 형식으로 변환 (ISO 8601 형식에 맞게 파싱)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# 시간대별 데이터 집계를 위한 시간 단위 설정 (예: 시간 단위로 그룹화)
df['hourly_timestamp'] = df['timestamp'].dt.floor('H')

# 시간대별 댓글 수 확인
hourly_counts = df.groupby('hourly_timestamp').size()
print("\n시간대별 댓글 수:")
print(hourly_counts)

# 3.6 텍스트 길이 계산 (text_length 컬럼 생성)
df['text_length'] = df['cleaned_content'].apply(len)

# 4. 데이터 저장
cleaned_data_path = "cleaned_data.csv"
df.to_csv(cleaned_data_path, index=False, encoding='utf-8-sig')
print(f"\n정제된 데이터를 {cleaned_data_path}에 저장했습니다.")

# 5. 간단한 데이터 탐색
# 플랫폼별 댓글 수
platform_counts = df.groupby('platform').size()
print("\n플랫폼별 댓글 수:")
print(platform_counts)

# 종목별 댓글 수
stock_counts = df.groupby('stock_name').size()
print("\n종목별 댓글 수:")
print(stock_counts)

# 텍스트 길이 분포 확인
print("\n텍스트 길이 통계:")
print(df['text_length'].describe())
