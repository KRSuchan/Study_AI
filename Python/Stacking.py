from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
raw_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    raw_data.data, raw_data.target, test_size=0.2, random_state=42, stratify=raw_data.target)
estimators = [('rf', RandomForestClassifier(n_estimators=10, random_state=42)),
              ('svc', SVC(kernel="poly", degree=3, C=10))]
clf = StackingClassifier(estimators=estimators,
                         final_estimator=LogisticRegression())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(X_test, y_test))
