@echo off
echo 🚀 Step 1: 데이터 전처리...
python label\prepare_data.py

echo 🚀 Step 2: 자동 감성 레이블링...
python label\auto_labeling.py

echo 🚀 Step 3: 모델 학습...
python label\train_model.py

echo ✅ 모든 작업 완료!
pause
