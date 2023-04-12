from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import fetch_openml

digits = fetch_openml('mnist_784', version=1, cache=True)

# from sklearn.datasets import load_digits
# digits = load_digits()

X, y = digits["data"], digits["target"]  # digits.data, digits.target
print(X.shape)  # (70000, 784)
print(y.shape)  # (70000,)

# 학습과 테스트 셋 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=30)

# 모델 학습
gnb_clf = GaussianNB()  # 불리안 데이터 : BernoulliNB, 발생 빈도 데이터 : MultinomialNB
gnb_clf.fit(X_train, y_train)

# 테스트 데이터에 대한 예측
y_pred = gnb_clf.predict(X_test)

# 정확도 계산
accuracy = 100.0 * (y_test == y_pred).sum() / X_test.shape[0]
print("Accuracy: ", round(accuracy, 2), "%")

# 또는 accuracy_score 함수 이용
# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test, y_pred)
