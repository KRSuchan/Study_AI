import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

# 학습 데이터 로드
input_file = 'data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# 학습 데이터 학습셋과 테스트셋으로 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

# 특징 선별을 위한 모델 정의
model = RandomForestClassifier()

# !! 특징 선택 시작 !!
fs = SelectFromModel(model, max_features=5)

# 특징 중요도를 계산하고 이를 바탕으로 특징을 선택하기 위해 원 데이터를 이용하여 학습
fs.fit(X_train, y_train)
# 선택된 특징에 맞추어 학습 데이터를 가공, 즉 선택된 특징에 해당하는 부분만 추출
X_train_fs = fs.transform(X_train)
X_test_fs = fs.transform(X_test)

# !! 특징 선택 끝 !!

# 변환된 학습 데이터를 이용하여 모델을 학습. 이 때 모델은 특징 추출을 위해 만든 # 모델을 사용해도 되고 새로운 모델 (예: LogisticRegression)을  사용해도 됨
# model = LogisticRegression(solver='liblinear')
model.fit(X_train_fs, y_train)

# 테스트셋에 대한 예측
y_pred = model.predict(X_test_fs)

# 예측 결과 성능 평가
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))
