from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris = load_iris()
X = iris.data[:, 2:]  # petal length and width
y = iris.target

# 학습과 테스트 셋 분리
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=30)
# 트리 분류기 생성 및 학습
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X_train, y_train)
# 테스트 데이터에 대한 예측
y_pred = tree_clf.predict(X_test)

# 정확도 계산
print(accuracy_score(y_test, y_pred))

print(tree_clf.predict_proba([[5, 1.5]]))
print(tree_clf.predict([[5, 1.5]]))