from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
raw_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    raw_data.data, raw_data.target, test_size=0.2, random_state=42, stratify=raw_data.target)
ada_clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=200,
                             algorithm="SAMME.R", learning_rate=0.5)
ada_clf.fit(X_train, y_train)
y_pred = ada_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
