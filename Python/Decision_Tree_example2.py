from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_digits

digits = load_digits()
X, y = digits.data, digits.target
print(X.shape)
print(y.shape)

# 학습과 테스트 셋 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=30)

# 트리 분류기 생성 및 학습
tree_clf = DecisionTreeClassifier(criterion="entropy", max_depth=10)
tree_clf.fit(X_train, y_train)
# 테스트 데이터에 대한 예측
y_pred = tree_clf.predict(X_test)

# 정확도 계산
print(accuracy_score(y_test, y_pred))
