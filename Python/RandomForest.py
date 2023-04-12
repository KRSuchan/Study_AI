from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
raw_data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    raw_data.data, raw_data.target, test_size=0.2, random_state=42, stratify=raw_data.target)
rnd_clf = RandomForestClassifier(
    n_estimators=500, max_leaf_nodes=50, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
y_pred = rnd_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
